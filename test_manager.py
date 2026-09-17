from graph.workflow import build_workflow


def main():
    graph = build_workflow()

    initial_state = {
        "company": "NVIDIA Corporation",
        "ticker": "NVDA",
        "research_question": (
            "Analyze NVIDIA's financial and market position."
        ),
    }

    result = graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("RESEARCH PLAN")
    print("=" * 60)

    print(
        result.get(
            "research_plan",
            "DATA NOT AVAILABLE",
        )
    )


if __name__ == "__main__":
    main()
