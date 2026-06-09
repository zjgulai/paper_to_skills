"""
Sandlock — 轻量 Agent 执行沙箱
arXiv: 2605.26298 | 2026年5月 | GitHub: github.com/multikernel/sandlock

Python 模拟层：SandboxPolicy + SandlockExecutor + ReversibleEffect + HTTPACLChecker
生产环境中需调用真实 Sandlock Rust 二进制；此模块提供开发/测试阶段的语义等价模拟。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import re


# ─────────────────────────────────────────────
# 数据类定义
# ─────────────────────────────────────────────

class Verdict(str, Enum):
    """沙箱执行判决结果"""
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    ROLLBACK = "ROLLBACK"


@dataclass
class HTTPACLRule:
    """HTTP 级别访问控制规则（Method + Host + 路径前缀）"""
    method: str          # "GET" / "POST" / "DELETE" / "*" (通配)
    host: str            # 允许访问的 Host，支持简单通配 "*.amazon.com"
    path_prefix: str     # 路径前缀，如 "/api/v1/inventory"

    def matches(self, method: str, host: str, path: str) -> bool:
        method_ok = self.method == "*" or self.method.upper() == method.upper()
        # 简单通配：*.domain.com
        if self.host.startswith("*."):
            domain = self.host[2:]
            host_ok = host == domain or host.endswith("." + domain)
        else:
            host_ok = self.host == host
        path_ok = path.startswith(self.path_prefix)
        return method_ok and host_ok and path_ok


@dataclass
class SandboxPolicy:
    """Sandlock 沙箱策略（静态层：Landlock + seccomp-bpf）"""
    readable_paths: list[str] = field(default_factory=list)   # 只读挂载白名单
    writable_paths: list[str] = field(default_factory=list)   # 可写路径白名单
    allowed_tcp_ports: list[int] = field(default_factory=lambda: [443, 80])
    http_acl_rules: list[HTTPACLRule] = field(default_factory=list)
    max_execution_seconds: int = 30      # 最大执行时间（秒）
    allow_network: bool = True           # 是否允许网络访问（总开关）

    def is_path_readable(self, path: str) -> bool:
        return any(path.startswith(p) for p in self.readable_paths)

    def is_path_writable(self, path: str) -> bool:
        return any(path.startswith(p) for p in self.writable_paths)

    def is_port_allowed(self, port: int) -> bool:
        return port in self.allowed_tcp_ports


@dataclass
class FileWriteRecord:
    """可逆文件系统：单次写操作记录"""
    original_path: str
    backup_path: Optional[str]   # None 表示该文件原本不存在
    was_created: bool            # True = 新建文件（回滚时删除）


@dataclass
class CommandResult:
    """沙箱命令执行结果"""
    command: str
    verdict: Verdict
    exit_code: int
    stdout: str
    stderr: str
    rollback_applied: bool
    policy_violations: list[str] = field(default_factory=list)
    execution_ms: float = 0.0


# ─────────────────────────────────────────────
# HTTP ACL 检查器
# ─────────────────────────────────────────────

class HTTPACLChecker:
    """HTTP 级别访问控制检查（Method/Host/路径细粒度）"""

    def __init__(self, rules: list[HTTPACLRule]):
        self.rules = rules

    def check(self, method: str, host: str, path: str) -> tuple[bool, str]:
        """
        检查 HTTP 请求是否允许。
        返回 (allowed: bool, reason: str)
        """
        for rule in self.rules:
            if rule.matches(method, host, path):
                return True, f"Matched rule: {rule.method} {rule.host}{rule.path_prefix}"
        return False, f"No ACL rule allows: {method} {host}{path}"

    def check_url(self, method: str, url: str) -> tuple[bool, str]:
        """从完整 URL 解析 host + path 并检查"""
        # 简单解析：https://host/path
        m = re.match(r"https?://([^/]+)(.*)", url)
        if not m:
            return False, f"Invalid URL format: {url}"
        host, path = m.group(1), m.group(2) or "/"
        return self.check(method, host, path)


# ─────────────────────────────────────────────
# 可逆文件系统效果
# ─────────────────────────────────────────────

class ReversibleEffect:
    """
    可逆文件系统效果（Copy-on-Write 快照模拟）。
    写操作先记录到 journal，成功则 commit，失败则 rollback。
    """

    def __init__(self, writable_paths: list[str]):
        self.writable_paths = writable_paths
        self._journal: list[FileWriteRecord] = []
        self._backup_dir = tempfile.mkdtemp(prefix="sandlock_cow_")

    def _is_allowed(self, path: str) -> bool:
        return any(path.startswith(p) for p in self.writable_paths)

    def record_write(self, target_path: str) -> bool:
        """
        在执行写操作前调用，记录原始状态。
        返回 False 表示路径不在可写白名单，拒绝写入。
        """
        if not self._is_allowed(target_path):
            return False
        if os.path.exists(target_path):
            backup = os.path.join(self._backup_dir, target_path.replace("/", "_"))
            shutil.copy2(target_path, backup)
            self._journal.append(FileWriteRecord(target_path, backup, was_created=False))
        else:
            self._journal.append(FileWriteRecord(target_path, None, was_created=True))
        return True

    def commit(self) -> None:
        """执行成功：清理备份，journal 归档"""
        shutil.rmtree(self._backup_dir, ignore_errors=True)
        self._journal.clear()

    def rollback(self) -> list[str]:
        """
        执行失败：恢复所有已记录写操作的原始状态。
        返回回滚的文件路径列表。
        """
        rolled_back = []
        for record in reversed(self._journal):
            if record.was_created:
                # 新建文件：删除
                if os.path.exists(record.original_path):
                    os.remove(record.original_path)
            else:
                # 覆盖文件：从备份还原
                if record.backup_path and os.path.exists(record.backup_path):
                    os.makedirs(os.path.dirname(record.original_path), exist_ok=True)
                    shutil.copy2(record.backup_path, record.original_path)
            rolled_back.append(record.original_path)
        shutil.rmtree(self._backup_dir, ignore_errors=True)
        self._journal.clear()
        return rolled_back

    def __del__(self):
        shutil.rmtree(self._backup_dir, ignore_errors=True)


# ─────────────────────────────────────────────
# Sandlock 执行器（模拟层）
# ─────────────────────────────────────────────

class SandlockExecutor:
    """
    Sandlock 沙箱执行器。
    生产环境：通过 subprocess 调用 Rust 二进制 `sandlock run --policy <json> -- <cmd>`
    测试/开发环境：此模拟层检查策略合规性，返回语义等价结果。
    """

    def __init__(self, policy: SandboxPolicy):
        self.policy = policy
        self.acl_checker = HTTPACLChecker(policy.http_acl_rules)
        self.reversible = ReversibleEffect(policy.writable_paths)

    def execute(self, command: str, simulate_http_calls: Optional[list[dict]] = None) -> CommandResult:
        """
        执行命令（模拟）。
        simulate_http_calls: [{"method": "GET", "url": "https://..."}] 模拟命令中的 HTTP 调用
        """
        import time
        start = time.time()
        violations: list[str] = []

        # 1. 检查网络访问
        if simulate_http_calls and not self.policy.allow_network:
            violations.append("网络访问被策略全局禁止")

        # 2. 检查 HTTP ACL
        if simulate_http_calls and self.policy.allow_network:
            for call in simulate_http_calls:
                allowed, reason = self.acl_checker.check_url(call["method"], call["url"])
                if not allowed:
                    violations.append(f"HTTP ACL 拦截: {reason}")

        # 3. 如果有违规，BLOCK 并回滚
        if violations:
            rolled = self.reversible.rollback()
            rollback_applied = len(rolled) > 0
            return CommandResult(
                command=command,
                verdict=Verdict.BLOCK,
                exit_code=1,
                stdout="",
                stderr="\n".join(violations),
                rollback_applied=rollback_applied,
                policy_violations=violations,
                execution_ms=round((time.time() - start) * 1000, 2),
            )

        # 4. 模拟正常执行（生产环境中此处调用真实 Rust 二进制）
        self.reversible.commit()
        return CommandResult(
            command=command,
            verdict=Verdict.ALLOW,
            exit_code=0,
            stdout=f"[sandlock] 命令在沙箱中成功执行: {command}",
            stderr="",
            rollback_applied=False,
            policy_violations=[],
            execution_ms=round((time.time() - start) * 1000 + 5.0, 2),  # 模拟 5ms 沙箱开销
        )

    def execute_with_file_write(
        self,
        command: str,
        target_file: str,
        content: str,
        should_fail: bool = False,
    ) -> CommandResult:
        """
        模拟带文件写入的命令执行，测试可逆文件系统效果。
        should_fail=True 时触发回滚验证。
        """
        import time
        start = time.time()

        # 记录写操作
        allowed = self.reversible.record_write(target_file)
        if not allowed:
            return CommandResult(
                command=command,
                verdict=Verdict.BLOCK,
                exit_code=1,
                stdout="",
                stderr=f"文件写入路径不在白名单: {target_file}",
                rollback_applied=False,
                policy_violations=[f"写路径拒绝: {target_file}"],
                execution_ms=round((time.time() - start) * 1000, 2),
            )

        # 执行写入
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        with open(target_file, "w") as f:
            f.write(content)

        if should_fail:
            # 模拟执行失败，触发回滚
            rolled = self.reversible.rollback()
            return CommandResult(
                command=command,
                verdict=Verdict.ROLLBACK,
                exit_code=1,
                stdout="",
                stderr="命令执行失败，文件写入已回滚",
                rollback_applied=True,
                policy_violations=["执行失败触发回滚"],
                execution_ms=round((time.time() - start) * 1000, 2),
            )

        self.reversible.commit()
        return CommandResult(
            command=command,
            verdict=Verdict.ALLOW,
            exit_code=0,
            stdout=f"文件成功写入: {target_file}",
            stderr="",
            rollback_applied=False,
            execution_ms=round((time.time() - start) * 1000 + 5.0, 2),
        )


# ─────────────────────────────────────────────
# 测试：WF-A 补货 Agent 沙箱场景
# ─────────────────────────────────────────────

def _test_wfa_replenishment_sandbox():
    """测试 WF-A 补货 Agent 沙箱：验证策略拦截和可逆文件系统回滚"""

    # 补货 Agent 策略：只能访问 ERP 特定接口
    policy = SandboxPolicy(
        readable_paths=["/tmp/inventory_data"],
        writable_paths=["/tmp/po_draft"],
        allowed_tcp_ports=[443],
        http_acl_rules=[
            HTTPACLRule(method="GET",  host="erp.company.com", path_prefix="/api/v1/inventory"),
            HTTPACLRule(method="POST", host="erp.company.com", path_prefix="/api/v1/purchase-orders"),
        ],
    )
    executor = SandlockExecutor(policy)

    # 测试 1：合法补货请求 → ALLOW
    result = executor.execute(
        "python generate_po.py --sku B001 --qty 500",
        simulate_http_calls=[
            {"method": "GET",  "url": "https://erp.company.com/api/v1/inventory/B001"},
            {"method": "POST", "url": "https://erp.company.com/api/v1/purchase-orders"},
        ],
    )
    assert result.verdict == Verdict.ALLOW, f"期望 ALLOW，得到 {result.verdict}"
    assert result.exit_code == 0
    print(f"[✓] 合法补货请求: {result.verdict} ({result.execution_ms:.1f}ms)")

    # 测试 2：越权访问财务系统 → BLOCK
    result = executor.execute(
        "python export_financials.py",
        simulate_http_calls=[
            {"method": "GET", "url": "https://finance.company.com/api/reports/profit"},
        ],
    )
    assert result.verdict == Verdict.BLOCK, f"期望 BLOCK，得到 {result.verdict}"
    assert "HTTP ACL 拦截" in result.stderr
    print(f"[✓] 越权财务访问被拦截: {result.verdict}")

    # 测试 3：可逆文件系统 — 执行失败时回滚
    import tempfile, os
    tmp_dir = "/tmp/po_draft"
    os.makedirs(tmp_dir, exist_ok=True)
    target = os.path.join(tmp_dir, "po_test.json")

    result = executor.execute_with_file_write(
        command="python write_po.py",
        target_file=target,
        content='{"sku": "B001", "qty": 500}',
        should_fail=True,
    )
    assert result.verdict == Verdict.ROLLBACK
    assert result.rollback_applied is True
    assert not os.path.exists(target), "回滚后文件应已删除"
    print(f"[✓] 可逆文件系统回滚: {result.verdict}, 文件已恢复")

    # 测试 4：WF-D 选品 Agent — 网络访问限制
    selection_policy = SandboxPolicy(
        readable_paths=["/tmp/competitor_data"],
        writable_paths=["/tmp/analysis_output"],
        allowed_tcp_ports=[443],
        http_acl_rules=[
            HTTPACLRule(method="GET", host="api.amazon.com",             path_prefix="/"),
            HTTPACLRule(method="GET", host="sellercentral.amazon.com",   path_prefix="/"),
        ],
    )
    sel_executor = SandlockExecutor(selection_policy)

    # 合法：访问 Amazon API
    result = sel_executor.execute(
        "python analyze_competitors.py",
        simulate_http_calls=[{"method": "GET", "url": "https://api.amazon.com/products/B002"}],
    )
    assert result.verdict == Verdict.ALLOW
    print(f"[✓] 选品 Agent Amazon 访问: {result.verdict}")

    # 非法：数据外泄到第三方
    result = sel_executor.execute(
        "python send_data.py",
        simulate_http_calls=[{"method": "POST", "url": "https://competitor-spy.io/upload"}],
    )
    assert result.verdict == Verdict.BLOCK
    print(f"[✓] 选品 Agent 数据外泄被拦截: {result.verdict}")

    print("\n[✓] Sandlock 沙箱全部测试通过（4/4）")


if __name__ == "__main__":
    _test_wfa_replenishment_sandbox()
