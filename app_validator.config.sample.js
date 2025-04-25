module.exports = {
  apps: [{
    name: "your_validator_name", // update this name
    script: "neurons/validator/validator.py",
    interpreter: "/root/miniconda3/envs/sn35-env/bin/python", // update this path
    env: {
        PYTHONPATH: './:${PYTHONPATH}',
        OPENAI_API_KEY: "your_openai_key",
        USE_TORCH: 1,
        VALIDATOR_USERNAME: "datapool_username",
        VALIDATOR_PASSWORD: "datapool_password",
        TASK_POOL_URL: "server_datapool_endpoint"
    },
    args: [
        "--netuid", "35",
        "--wallet.name", "your_wallet_name",
        "--wallet.hotkey", "your_hotkey",
        "--subtensor.network", "finney",
        "--neuron_type", "validator",
        "--llm_client.gpt_url", "https://api.openai.com/v1",
        "--llm_client.gpt_model", "gpt-4o-mini",
        "--llm_client.vllm_url", "your_vllm_endpoint",
        "--llm_client.vllm_model", "your_vllm_model_name",
        "--llm_client.vllm_key", "your_vllm_key",
        "--wandb.project_name", "logicnet-mainnet",
        "--logging.debug",
    ],
    autorestart: true,
  }]
};
