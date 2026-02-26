#!/usr/bin/env python3
"""
Turing — 数学研究智能体系统入口。

用法::

    python main.py                    # 交互模式
    python main.py --task "证明..."    # 单任务模式
    python main.py --train 60         # 训练模式（分钟）
    python main.py --status           # 系统状态检查

更多信息参见 README.md。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
from pathlib import Path

from loguru import logger

from turing.config import get_config, TuringConfig
from turing.agents.turing_agent import TuringAgent
from turing.utils import setup_logging


# ---------- CLI modes -------------------------------------------------------

async def interactive_mode(agent: TuringAgent):
    """交互式命令行模式。"""
    print("\n" + "=" * 50)
    print("  Turing 数学研究智能体 — 交互模式")
    print("  输入数学任务，或使用以下命令:")
    print("    /train [分钟]   进入训练模式")
    print("    /status         显示系统状态")
    print("    /reflect        触发反思")
    print("    /evaluate       系统全面评估")
    print("    /evolve         评估 + 演化流程")
    print("    /optimize       系统优化")
    print("    /quit           退出")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input("Turing> ").strip()
            )
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.split()
            cmd_name = cmd[0].lower()

            if cmd_name == "/quit":
                break
            elif cmd_name == "/train":
                mins = int(cmd[1]) if len(cmd) > 1 else 30
                await agent.enter_training_mode(duration_minutes=mins)
            elif cmd_name == "/status":
                snapshot = agent.resource_manager.assess()
                print(agent._generate_status_report(
                    snapshot,
                    await agent.lean.check_status(),
                    agent.ltm.get_stats(),
                    agent.pm.get_task_count(),
                    agent.pm.get_reflection_count(),
                ))
            elif cmd_name == "/reflect":
                result = await agent.reflection.reflect()
                print(f"反思结果: {result.get('report', '无')[:500]}")
            elif cmd_name == "/evaluate":
                print("正在评估系统...")
                result = await agent.evaluate_system()
                if result:
                    print(f"\n评估摘要:\n{result.get('summary', '无')}")
                    proposals = result.get('proposals', [])
                    if proposals:
                        print(f"\n改进建议 ({len(proposals)} 条):")
                        for i, p in enumerate(proposals[:5], 1):
                            prio = p.get('priority', '?')
                            desc = p.get('description', '?')[:100]
                            print(f"  {i}. [{prio}] {desc}")
                    print(f"\n建议演化: {'是' if result.get('should_evolve') else '否'}")
                else:
                    print("评估失败")
            elif cmd_name == "/evolve":
                print("正在执行 评估-演化 流程...")
                result = await agent.evaluate_and_evolve()
                if result:
                    print(f"\n评估摘要:\n{result.get('evaluation_summary', '无')}")
                    plan = result.get('evolution_plan', {})
                    phase = plan.get('evolution_phase', '未知')
                    print(f"\n演化主题: {phase}")
                    expected = plan.get('expected_improvements', {})
                    if expected:
                        print(f"预期改进: {expected}")
                    print(f"已执行: {'是' if result.get('executed') else '否'}")
                else:
                    print("演化流程失败")
            elif cmd_name == "/optimize":
                result = await agent.optimize_system()
                print(f"优化结果: {result}")
            else:
                print(f"未知命令: {cmd_name}")
        else:
            # 处理数学任务
            result = await agent.process_task(user_input)
            print(f"\n--- 任务结果 ---")
            success = result.get("success", False)
            print(f"状态: {'✓ 成功' if success else '✗ 失败'}")
            if result.get("lean_code"):
                print(f"Lean 代码:\n{result['lean_code']}")
            if result.get("error") or result.get("last_error"):
                print(f"错误: {result.get('error') or result.get('last_error')}")
            if result.get("response"):
                print(f"回复:\n{result['response'][:1000]}")
            if result.get("review"):
                review = result["review"]
                print(f"审查: 评分 {review.get('score','?')}/10 — {review.get('verdict','?')}")
            if result.get("evaluation"):
                print(f"评估: {result['evaluation']}")
            print("---\n")


async def single_task_mode(agent: TuringAgent, task: str):
    """执行单个任务，将结果以 JSON 形式输出后退出。"""
    result = await agent.process_task(task)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return result


async def training_mode(agent: TuringAgent, minutes: int):
    """训练模式。"""
    await agent.enter_training_mode(duration_minutes=minutes)


async def status_mode(agent: TuringAgent):
    """打印系统状态并退出。"""
    snapshot = agent.resource_manager.assess()
    lean_status = await agent.lean.check_status()
    ltm_stats = agent.ltm.get_stats()
    pm_tasks = agent.pm.get_task_count()
    pm_ref = agent.pm.get_reflection_count()
    print(agent._generate_status_report(snapshot, lean_status, ltm_stats, pm_tasks, pm_ref))


# ---------- main -----------------------------------------------------------

async def async_main(args: argparse.Namespace):
    config = get_config()
    setup_logging(
        log_dir=Path(config.system.data_dir) / "logs",
        log_prefix="turing",
        level=config.system.log_level,
    )

    agent = TuringAgent(config)

    # 注册关闭信号
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(agent.shutdown()))

    try:
        await agent.initialize()

        if args.status:
            await status_mode(agent)
        elif args.task:
            await single_task_mode(agent, args.task)
        elif args.train is not None:
            await training_mode(agent, args.train)
        elif args.loop:
            await agent.main_loop()
        else:
            await interactive_mode(agent)
    except Exception as e:
        logger.exception(f"致命错误: {e}")
    finally:
        await agent.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Turing — 数学研究智能体",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--task", "-t", type=str, help="直接执行的数学任务")
    parser.add_argument("--train", type=int, nargs="?", const=60, help="进入训练模式（分钟，默认60）")
    parser.add_argument("--status", "-s", action="store_true", help="显示系统状态并退出")
    parser.add_argument("--loop", "-l", action="store_true", help="进入自主运行循环")
    parser.add_argument("--config", "-c", type=str, default="config.yaml", help="配置文件路径")

    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
