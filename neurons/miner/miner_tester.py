import bittensor as bt
import httpx

from logicnet.miner.forward import solve
from logicnet.protocol import LogicSynapse


async def solve_task(synapse: LogicSynapse) -> LogicSynapse:
    try:
        bt.logging.info(f"Received synapse: {synapse}")
        logic_question: str = synapse.logic_question
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post("http://127.0.0.1:8000/solve_task", json={'logic_question':logic_question})
            data = response.json()
            synapse.logic_reasoning = data['logic_reasoning']
            synapse.logic_answer = data['logic_answer']

        bt.logging.info(f"Logic answer: {synapse.logic_answer}")
        bt.logging.info(f"Logic reasoning: {synapse.logic_reasoning}")
        return synapse
    except Exception as e:
        bt.logging.error(f"Error in forward: {e}")
        traceback.print_exc()



async def test():
    synapse = LogicSynapse(logic_question="calculate sin(30)")
    synapse = await solve_task(
        synapse=synapse
    )
    print(synapse)


import asyncio

# This is the main function, which runs the miner.
if __name__ == "__main__":
    asyncio.run(test())
