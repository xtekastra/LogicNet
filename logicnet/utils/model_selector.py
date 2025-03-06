import random

def model_selector(model_pool, task_type="create_task"):
    """
    Select the model based on the task
    Input:
        model_pool: dict
            The pool of models available
        task_type: str
            The type of task to be performed. Can be either "create_task" or "score_task"
    """
    if task_type == "create_task" and "vllm" in model_pool:
        model = model_pool["vllm"]
        base_url = model_pool["vllm_base_url"]
        api_key = model_pool["vllm_key"]
    else:
        model = model_pool["gpt"]
        base_url = model_pool["gpt_base_url"]
        api_key = model_pool["gpt_key"]
    return model, base_url, api_key