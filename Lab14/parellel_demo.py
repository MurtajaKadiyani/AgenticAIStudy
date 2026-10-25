import asyncio
import time


async def research_agent():

    print("Research Agent started")

    await asyncio.sleep(3)

    print("Research Agent finished")

    return "Research result"


async def technology_agent():

    print("Technology Agent started")

    await asyncio.sleep(2)

    print("Technology Agent finished")

    return "Technology result"


async def market_agent():

    print("Market Agent started")

    await asyncio.sleep(4)

    print("Market Agent finished")

    return "Market result"


async def run_sequential():

    start = time.perf_counter()

    research = await research_agent()

    technology = await technology_agent()

    market = await market_agent()

    elapsed = (
        time.perf_counter()
        - start
    )

    print("\nSequential Results:")

    print(research)
    print(technology)
    print(market)

    print(
        f"\nSequential time: "
        f"{elapsed:.2f} seconds"
    )


async def run_parallel():

    start = time.perf_counter()

    results = await asyncio.gather(
        research_agent(),
        technology_agent(),
        market_agent()
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print("\nParallel Results:")

    for result in results:

        print(result)

    print(
        f"\nParallel time: "
        f"{elapsed:.2f} seconds"
    )


async def main():

    print("=" * 50)
    print("SEQUENTIAL EXECUTION")
    print("=" * 50)

    await run_sequential()

    print("\n" + "=" * 50)
    print("PARALLEL EXECUTION")
    print("=" * 50)

    await run_parallel()


if __name__ == "__main__":

    asyncio.run(main())
