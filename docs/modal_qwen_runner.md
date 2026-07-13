# Qwen3-32B Modal runner

This runner provides live model generation and residual-stream extraction for
the shared Scenario 1 generator. It does not replace the Scenario 1 input
registry, agent chain, v2 schema, validator, or outcome adapter.

## What can be tested without the Scenario 1 dataset

The included request fixture checks:

- the Qwen model and thinking-mode configuration;
- the controlled decoding settings;
- request and message validation;
- the rule that only Worker1 may receive raw poison;
- the 64-layer activation range;
- separation of thinking content from the downstream message;
- safe activation artifact paths.

This fixture is only an infrastructure check. It is not a Scenario 1
trajectory and must not be included in the dataset or manifest.

Run the local tests without starting Modal compute:

```bash
python -m pytest tests/test_qwen_modal.py -q
```

Compile the Modal app without deploying it:

```bash
python -m py_compile modal_qwen_runner.py src/infrastructure/qwen_modal.py
```

## Modal resources

The app uses the shared `agileai` workspace and these resources:

- `spec-gap-qwen3-32b-model`: cached Qwen weights;
- `spec-gap-scenario1-artifacts`: activation tensors and metadata;
Qwen3-32B is public, so the initial runner does not require a Hugging Face
secret. If authenticated downloads are added later, create the secret in the
Modal dashboard. Do not commit a token or put it in trajectory files.

## Commands and cost boundary

Validate a request without calling a remote function or GPU:

```bash
modal run modal_qwen_runner.py \
  --request-path tests/fixtures/qwen_agent_turn_request.json \
  --action validate
```

Downloading the weights uses remote CPU, network, and Volume storage but does
not start a GPU:

```bash
modal run modal_qwen_runner.py \
  --request-path tests/fixtures/qwen_agent_turn_request.json \
  --action download
```

Do not run the download until the model revision and storage plan are approved.

An H200 starts only when both `--action run` and the confirmation string are
provided:

```bash
modal run modal_qwen_runner.py \
  --request-path path/to/real_agent_turn_request.json \
  --action run \
  --confirm-paid-run RUN_H200 \
  --output-path path/to/model_turn_output.json
```

The H200 path should be used only after a real shared match-group input passes
the Scenario 1 validator. A full scientific result still requires live
Scenario 1 inputs, downstream handoffs, explicit tool execution evidence, and
the shared adapter.
