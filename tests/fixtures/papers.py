"""Paper fixtures for testing - snapshot from production database.

These papers have dates stripped. At test time, dates are assigned dynamically
to simulate "today", "yesterday", etc. for testing date-aware queries.
"""

from datetime import date, timedelta

# Raw paper data from database (dates stripped)
PAPERS_RAW = [
    {
        "id": "2601.20614",
        "title": "Harder Is Better: Boosting Mathematical Reasoning via Difficulty-Aware GRPO and Multi-Aspect Question Reformulation",
        "summary": "Reinforcement Learning with Verifiable Rewards (RLVR) offers a robust mechanism for enhancing mathematical reasoning in large models. However, we identify a systematic lack of emphasis on more challenging questions in existing methods from both algorithmic and data perspectives, despite their importance for refining underdeveloped capabilities. Algorithmically, widely used Group Relative Policy Optimization (GRPO) suffers from an implicit imbalance where the magnitude of policy updates is lower for harder questions. Data-wise, augmentation approaches primarily rephrase questions to enhance diversity without systematically increasing intrinsic difficulty. To address these issues, we propose a two-dual MathForge framework to improve mathematical reasoning by targeting harder questions from both perspectives, which comprises a Difficulty-Aware Group Policy Optimization (DGPO) algorithm and a Multi-Aspect Question Reformulation (MQR) strategy.",
        "authors": ["Yanqi Dai", "Yuxiang Ji", "Xiao Zhang", "Yong Wang", "Xiangxiang Chu", "Zhiwu Lu"],
        "upvotes": 93,
    },
    {
        "id": "2601.20540",
        "title": "Advancing Open-source World Models",
        "summary": "We present LingBot-World, an open-sourced world simulator stemming from video generation. Positioned as a top-tier world model, LingBot-World offers the following features. (1) It maintains high fidelity and robust dynamics in a broad spectrum of environments, including realism, scientific contexts, cartoon styles, and beyond. (2) It enables a minute-level horizon while preserving contextual consistency over time, which is also known as 'long-term memory'. (3) It supports real-time interactivity, achieving a latency of under 1 second when producing 16 frames per second.",
        "authors": ["Robbyant Team", "Zelin Gao", "Qiuyu Wang"],
        "upvotes": 65,
    },
    {
        "id": "2601.19325",
        "title": "Innovator-VL: A Multimodal Large Language Model for Scientific Discovery",
        "summary": "We present Innovator-VL, a scientific multimodal large language model designed to advance understanding and reasoning across diverse scientific domains while maintaining excellent performance on general vision tasks. Contrary to the trend of relying on massive domain-specific pretraining and opaque pipelines, our work demonstrates that principled training design and transparent methodology can yield strong scientific intelligence with substantially reduced data requirements.",
        "authors": ["Zichen Wen", "Boxue Yang", "Shuang Chen"],
        "upvotes": 53,
    },
    {
        "id": "2601.20552",
        "title": "DeepSeek-OCR 2: Visual Causal Flow",
        "summary": "We present DeepSeek-OCR 2 to investigate the feasibility of a novel encoder-DeepEncoder V2-capable of dynamically reordering visual tokens upon image semantics. Conventional vision-language models (VLMs) invariably process visual tokens in a rigid raster-scan order (top-left to bottom-right) with fixed positional encoding when fed into LLMs. However, this contradicts human visual perception, which follows flexible yet semantically coherent scanning patterns driven by inherent logical structures.",
        "authors": ["Haoran Wei", "Yaofeng Sun", "Yukun Li"],
        "upvotes": 25,
    },
    {
        "id": "2601.20209",
        "title": "Spark: Strategic Policy-Aware Exploration via Dynamic Branching for Long-Horizon Agentic Learning",
        "summary": "Reinforcement learning has empowered large language models to act as intelligent agents, yet training them for long-horizon tasks remains challenging due to the scarcity of high-quality trajectories, especially under limited resources. Existing methods typically scale up rollout sizes and indiscriminately allocate computational resources among intermediate steps.",
        "authors": ["Jinyang Wu", "Shuo Yang", "Changpeng Yang"],
        "upvotes": 12,
    },
    {
        "id": "2601.20834",
        "title": "Linear representations in language models can change dramatically over a conversation",
        "summary": "Language model representations often contain linear directions that correspond to high-level concepts. Here, we study the dynamics of these representations: how representations evolve along these dimensions within the context of (simulated) conversations. We find that linear representations can change dramatically over a conversation.",
        "authors": ["Andrew Kyle Lampinen", "Yuxuan Li", "Eghbal Hosseini"],
        "upvotes": 8,
    },
    {
        "id": "2601.20802",
        "title": "Reinforcement Learning via Self-Distillation",
        "summary": "Large language models are increasingly post-trained with reinforcement learning in verifiable domains such as code and math. Yet, current methods for reinforcement learning with verifiable rewards (RLVR) learn only from a scalar outcome reward per attempt, creating a severe credit-assignment bottleneck. Many verifiable environments actually provide rich textual feedback, such as runtime errors or judge evaluations, that explain why an attempt failed.",
        "authors": ["Jonas Hübotter", "Frederike Lübeck", "Lejs Behric"],
        "upvotes": 5,
    },
    {
        "id": "2601.20055",
        "title": "VERGE: Formal Refinement and Guidance Engine for Verifiable LLM Reasoning",
        "summary": "Despite the syntactic fluency of Large Language Models (LLMs), ensuring their logical correctness in high-stakes domains remains a fundamental challenge. We present a neurosymbolic framework that combines LLMs with SMT solvers to produce verification-guided answers through iterative refinement.",
        "authors": ["Vikash Singh", "Darion Cassel", "Nathaniel Weir"],
        "upvotes": 5,
    },
    # Yesterday's papers
    {
        "id": "2601.18491",
        "title": "AgentDoG: A Diagnostic Guardrail Framework for AI Agent Safety and Security",
        "summary": "The rise of AI agents introduces complex safety and security challenges arising from autonomous tool use and environmental interactions. Current guardrail models lack agentic risk awareness and transparency in risk diagnosis. To introduce an agentic guardrail that covers complex and numerous risky behaviors, we first propose a unified three-dimensional taxonomy.",
        "authors": ["Dongrui Liu", "Qihan Ren", "Chen Qian"],
        "upvotes": 85,
    },
    {
        "id": "2601.18631",
        "title": "AdaReasoner: Dynamic Tool Orchestration for Iterative Visual Reasoning",
        "summary": "When humans face problems beyond their immediate capabilities, they rely on tools, providing a promising paradigm for improving visual reasoning in multimodal large language models (MLLMs). Effective reasoning, therefore, hinges on knowing which tools to use, when to invoke them, and how to compose them over multiple steps.",
        "authors": ["Mingyang Song", "Haoyu Sun", "Jiawei Gu"],
        "upvotes": 45,
    },
    {
        "id": "2601.18692",
        "title": "A Pragmatic VLA Foundation Model",
        "summary": "Offering great potential in robotic manipulation, a capable Vision-Language-Action (VLA) foundation model is expected to faithfully generalize across tasks and platforms while ensuring cost efficiency. To this end, we develop LingBot-VLA with around 20,000 hours of real-world data from 9 popular dual-arm robot configurations.",
        "authors": ["Wei Wu", "Fan Lu", "Yunnan Wang"],
        "upvotes": 41,
    },
    {
        "id": "2601.19798",
        "title": "Youtu-VL: Unleashing Visual Potential via Unified Vision-Language Supervision",
        "summary": "Despite the significant advancements represented by Vision-Language Models (VLMs), current architectures often exhibit limitations in retaining fine-grained visual information, leading to coarse-grained multimodal comprehension.",
        "authors": ["Zhixiang Wei", "Yi Li", "Zhehan Kan"],
        "upvotes": 30,
    },
    {
        "id": "2601.19834",
        "title": "Visual Generation Unlocks Human-Like Reasoning through Multimodal World Models",
        "summary": "Humans construct internal world models and reason by manipulating the concepts within these models. Recent advances in AI, particularly chain-of-thought (CoT) reasoning, approximate such human cognitive abilities, where world models are believed to be embedded within large language models.",
        "authors": ["Jialong Wu", "Xiaoying Zhang", "Hongyi Yuan"],
        "upvotes": 23,
    },
    {
        "id": "2601.17645",
        "title": "AVMeme Exam: A Multimodal Multilingual Multicultural Benchmark for LLMs' Contextual and Cultural Knowledge and Thinking",
        "summary": "Internet audio-visual clips convey meaning through time-varying sound and motion, which extend beyond what text alone can represent. To examine whether AI models can understand such signals in human cultural contexts, we introduce AVMeme Exam.",
        "authors": ["Xilin Jiang", "Qiaolin Wang", "Junkai Wu"],
        "upvotes": 22,
    },
    {
        "id": "2601.09150",
        "title": "World Craft: Agentic Framework to Create Visualizable Worlds via Text",
        "summary": "Large Language Models (LLMs) motivate generative agent simulation (e.g., AI Town) to create a 'dynamic world', holding immense value across entertainment and research. However, for non-experts, especially those without programming skills, it isn't easy to customize a visualizable environment by themselves.",
        "authors": ["Jianwen Sun", "Yukang Feng", "Kaining Ying"],
        "upvotes": 18,
    },
    {
        "id": "2601.19895",
        "title": "Post-LayerNorm Is Back: Stable, ExpressivE, and Deep",
        "summary": "Large language model (LLM) scaling is hitting a wall. Widening models yields diminishing returns, and extending context length does not improve fundamental expressivity. In contrast, depth scaling offers theoretically superior expressivity, yet current Transformer architectures struggle to train reliably at extreme depths.",
        "authors": ["Chen Chen", "Lai Wei"],
        "upvotes": 15,
    },
]


def get_papers_for_date(target_date: date) -> list[dict]:
    """Get papers with dates assigned based on relative day.

    First 8 papers -> today
    Next 8 papers -> yesterday
    """
    today = date.today()
    yesterday = today - timedelta(days=1)

    papers = []
    for i, paper in enumerate(PAPERS_RAW):
        paper_copy = paper.copy()
        # First 8 are "today", next 8 are "yesterday"
        paper_copy["date"] = today if i < 8 else yesterday
        paper_copy["published_at"] = paper_copy["date"].isoformat()

        if paper_copy["date"] == target_date:
            papers.append(paper_copy)

    return sorted(papers, key=lambda p: p["upvotes"], reverse=True)


def get_available_dates(days_back: int = 7) -> list[dict]:
    """Get available dates with paper counts."""
    today = date.today()
    yesterday = today - timedelta(days=1)

    return [
        {"date": today.isoformat(), "count": 8},
        {"date": yesterday.isoformat(), "count": 8},
    ]


def get_papers_for_ranking(target_date: date) -> dict[str, dict]:
    """Get papers in format needed by ranking agent."""
    papers = get_papers_for_date(target_date)
    return {
        p["id"]: {"title": p["title"], "summary": p["summary"]}
        for p in papers
    }


def get_top_papers(target_date: date, limit: int = 10) -> list[dict]:
    """Get top N papers by upvotes for a date."""
    papers = get_papers_for_date(target_date)
    return papers[:limit]


def get_paper_stats(days_back: int = 7) -> dict:
    """Get aggregate statistics about papers."""
    today = date.today()
    yesterday = today - timedelta(days=1)

    return {
        "total_papers": 16,
        "days_with_papers": 2,
        "earliest_date": yesterday.isoformat(),
        "latest_date": today.isoformat(),
        "avg_upvotes": 35.0,
    }
