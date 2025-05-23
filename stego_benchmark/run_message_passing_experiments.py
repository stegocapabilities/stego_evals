import asyncio
from dataclasses import dataclass
from pathlib import Path

from runner import ExperimentRunner

from stego_benchmark.config import (
    BenchmarkConfig,
    DefinedEncodingScheme,
    EncodeAndDecodeConfig,
    EncodeBaselineConfig,
    ExperimentConfig,
    ExperimentMode,
    LLMConfig,
    MessageTaskType,
    MessageType,
    MonitorConfig,
    MonitorType,
    SimpleEncodeOrDecodeConfig,
    message_task_type_mapping,
    message_type_mapping,
)


async def run_encode_and_decode_experiment(
    model,
    encoding_scheme,
    cot,
    model_temperature: float = 1.0,
    num_samples: int = 300,
    input_message_type: MessageType = MessageType.SHORT_PHRASE,
    task_type: MessageTaskType = MessageTaskType.CONVERSATIONAL,
    output_dir: str = "results/encode_and_decode/",
    monitors: list[MonitorConfig] = [],
):

    if model == "anthropic/claude-3-7-sonnet-extended":
        model_config = LLMConfig(
            model="anthropic/claude-3-7-sonnet-20250219", temperature=model_temperature, extended_thinking_tokens=4096
        )
    else:
        model_config = LLMConfig(model=model, temperature=model_temperature)

    input_message_path = Path(f"datasets/message_data/{message_type_mapping[input_message_type][0]}")
    if task_type == MessageTaskType.NO_TASK:
        task_dataset_path = None
    else:
        task_dataset_path = Path(f"datasets/task_data/{message_task_type_mapping[task_type]}")

    experiment_mode = ExperimentMode.ENCODE_AND_DECODE
    setting_config = EncodeAndDecodeConfig(
        message_dataset_path=input_message_path,
        message_type=input_message_type,
        task_dataset_path=task_dataset_path,
        encoding_scheme=encoding_scheme,
        include_cot=cot,
        monitor_configs=monitors,
    )

    experiment_config = ExperimentConfig(
        experiment_mode=experiment_mode,
        model=model_config,
        dataset_path=input_message_path,
        output_dir=Path(output_dir),
        num_samples=num_samples,
        setting_config=setting_config,
        verbose=False,
        batch_size=100 if "openai" in model else 20,
    )

    benchmark_config = BenchmarkConfig(experiment=experiment_config)

    runner = ExperimentRunner(benchmark_config)

    print(f"Running {model} with {encoding_scheme} and {cot}")
    results = await runner.run()


async def run_encode_baseline_experiment(
    model,
    model_temperature: float = 1.0,
    num_samples: int = 300,
    task_dataset_path: str = "datasets/anthropic_helpful_test.jsonl",
    output_dir: str = "results/encode_baseline/",
    monitors: list[MonitorConfig] = [],
):
    experiment_mode = ExperimentMode.ENCODE_BASELINE

    setting_config = EncodeBaselineConfig(task_dataset_path=Path(task_dataset_path), monitor_configs=monitors)
    experiment_config = ExperimentConfig(
        experiment_mode=experiment_mode,
        model=LLMConfig(model=model, temperature=model_temperature),
        dataset_path=Path(task_dataset_path),
        output_dir=Path(output_dir),
        num_samples=num_samples,
        setting_config=setting_config,
        verbose=False,
        batch_size=100 if "openai" in model else 20,
    )

    benchmark_config = BenchmarkConfig(experiment=experiment_config)

    runner = ExperimentRunner(benchmark_config)

    results = await runner.run()


async def run_encode_and_decode_experiments(
    output_dir: str,
    monitors: list[MonitorConfig] = [],
    num_samples: int = 300,
    encoding_schemes: list[DefinedEncodingScheme] = [],
    models: list[str] = [],
    cot_list: list[bool] = [],
    input_message_type: MessageType = MessageType.SHORT_PHRASE,
    task_type: MessageTaskType = MessageTaskType.CONVERSATIONAL,
):
    models_w_cot_required = ["openai/o3-mini-2025-01-31", "anthropic/claude-3-7-sonnet-extended"]

    model_tasks = {model: [] for model in models}

    for model in models:
        for encoding_scheme in encoding_schemes:
            for cot in cot_list:
                if model not in models_w_cot_required or cot:
                    # Create the task but don't group it by model
                    task = run_encode_and_decode_experiment(
                        model,
                        encoding_scheme,
                        cot,
                        num_samples=num_samples,
                        output_dir=output_dir + encoding_scheme.value + "/",
                        monitors=monitors,
                        input_message_type=input_message_type,
                        task_type=task_type,
                    )
                    model_tasks[model].append(task)

    # For each model, create a task that runs all its experiments sequentially
    async def run_model_tasks(tasks):
        results = []
        for task in tasks:
            result = await task
            results.append(result)
        return results

    # Create one task per model that will run all that model's experiments in sequence
    model_runners = [run_model_tasks(tasks) for model, tasks in model_tasks.items()]

    # Run each model's sequence of tasks concurrently with other models
    all_results = await asyncio.gather(*model_runners)


all_models = [
    "openai/gpt-4o-2024-08-06",
    "openai/gpt-4.1-2025-04-14",
    "openai/gpt-4.5-preview",
    "anthropic/claude-3-5-sonnet-20241022",
    "anthropic/claude-3-7-sonnet-20250219",
    "openai/o3-mini-2025-01-31",
]

all_but_gpt_45 = [model for model in all_models if model != "openai/gpt-4.5-preview"]

cot_vals = [False, True]


async def run_encode_and_decode_experiments_41_and_42():
    output_dir = "results/encode_and_decode_41_and_42/"

    encoding_schemes = [
        DefinedEncodingScheme.SENTENCE_LETTER,
        DefinedEncodingScheme.ALT_SENTENCE_LETTER,
        DefinedEncodingScheme.THIRD_WORD_LETTER,
        DefinedEncodingScheme.ROT13_THIRD_WORD_LETTER,
        DefinedEncodingScheme.CUSTOM_CIPHER,
    ]

    num_samples = 400

    monitor_types = [MonitorType.COT_NUMERIC2]
    monitors = [
        MonitorConfig(model="openai/gpt-4.1-mini-2025-04-14", temperature=1.0, monitor_type=monitor, top_logprobs=5)
        for monitor in monitor_types
    ]

    await run_encode_and_decode_experiments(output_dir, monitors, num_samples, encoding_schemes, all_models, cot_vals)


async def run_encode_and_decode_experiments_43():
    output_dir = "results/encode_and_decode_43/"

    encoding_schemes = [
        DefinedEncodingScheme.SENTENCE_LETTER,
        DefinedEncodingScheme.ALT_SENTENCE_LETTER,
        DefinedEncodingScheme.THIRD_WORD_LETTER,
        DefinedEncodingScheme.ROT13_THIRD_WORD_LETTER,
        DefinedEncodingScheme.ROTN,
        DefinedEncodingScheme.BASE64,
        DefinedEncodingScheme.HEX,
        DefinedEncodingScheme.BINARY8,
        DefinedEncodingScheme.BASE64_MULTIPLE,
        DefinedEncodingScheme.HEX_MULTIPLE,
        DefinedEncodingScheme.SUBSTITUTION_MULTIPLE,
        DefinedEncodingScheme.SUBSTITUTION,
        DefinedEncodingScheme.HEX_ROT1,
        DefinedEncodingScheme.ROTN_WITH_BARS,
        DefinedEncodingScheme.BINARY8_SWAP,
    ]

    monitor_types = [MonitorType.COT_NUMERIC2]
    monitors = [
        MonitorConfig(model="openai/gpt-4.1-mini-2025-04-14", temperature=1.0, monitor_type=monitor, top_logprobs=5)
        for monitor in monitor_types
    ]
    num_samples = 100

    await run_encode_and_decode_experiments(output_dir, monitors, num_samples, encoding_schemes, all_models, cot_vals)


async def run_appendix_datasets():
    output_dir = "results/appendix_datasets/"
    encoding_schemes = [DefinedEncodingScheme.CUSTOM_CIPHER]
    num_samples = 400
    monitor_types = [MonitorType.COT_NUMERIC2]
    monitors = [
        MonitorConfig(model="openai/gpt-4.1-mini-2025-04-14", temperature=1.0, monitor_type=monitor, top_logprobs=5)
        for monitor in monitor_types
    ]
    task_types = [MessageTaskType.ANYTHING, MessageTaskType.CODING, MessageTaskType.AQUA]
    for task_type in task_types:
        await run_encode_and_decode_experiments(
            output_dir + task_type.value + "/",
            monitors,
            num_samples,
            encoding_schemes,
            all_but_gpt_45,
            cot_vals,
            task_type=task_type,
        )


async def run_appendix_hidden_messages():
    output_dir = "results/appendix_hidden_messages/"
    encoding_schemes = [DefinedEncodingScheme.CUSTOM_CIPHER]
    num_samples = 400
    monitor_types = [MonitorType.COT_NUMERIC2]
    monitors = [
        MonitorConfig(model="openai/gpt-4.1-mini-2025-04-14", temperature=1.0, monitor_type=monitor, top_logprobs=5)
        for monitor in monitor_types
    ]
    input_message_types = [MessageType.SENTENCES, MessageType.SITUATION, MessageType.SHORT_CHARS]
    for input_message_type in input_message_types:
        await run_encode_and_decode_experiments(
            output_dir + input_message_type.value + "/",
            monitors,
            num_samples,
            encoding_schemes=encoding_schemes,
            models=all_but_gpt_45,
            cot_list=cot_vals,
            input_message_type=input_message_type,
        )


if __name__ == "__main__":
    asyncio.run(run_encode_and_decode_experiments_41_and_42())
    asyncio.run(run_encode_and_decode_experiments_43())
    asyncio.run(run_appendix_hidden_messages())
    asyncio.run(run_appendix_datasets())
