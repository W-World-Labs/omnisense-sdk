# OmniSense Python SDK

Version `1.2.6` requires Python 3.11 or newer, matching the pinned Pillow 13.0.0 dependency. This metadata correction leaves the SDK interfaces and behavior unchanged.

This source subset provides a Python image-task SDK and CLI for an independent, user-configured HTTP engine. It does not contain a Rust service, native inference implementation, model implementation, speech worker, engine package, model weights, datasets or production deployment configuration. Distribution name: `omnisense-sdk`; import and command name: `omnisense`. No public package-index release is claimed by this source snapshot.

## Install and use

```bash
python -m pip install .
export OMNISENSE_ENGINE_URL__OBJECT_DETECTION=http://127.0.0.1:19090
export OMNISENSE_ENGINE_MODEL__OBJECT_DETECTION=my-detector
# Inject OMNISENSE_ENGINE_TOKEN__OBJECT_DETECTION securely if your engine requires it.
omnisense detect /absolute/user-owned/image.jpg
```

```python
import omnisense
result = omnisense.process_image("/absolute/user-owned/image.jpg", task="detect")
print(result["status"], result["outputs"])
```

An engine and a model you are entitled to use must already be installed and registered separately. This SDK never installs an engine, opens a model artifact or downloads missing weights. `auto_download=True` is rejected. No default supplier or implicit fallback exists. Missing engine configuration returns `missing_model`; failures return a structured response. `mock-detector` is a synthetic rectangle demonstration and has no model accuracy claim.

`process_image`, `process_images`, `detection.run`, `AlgorithmRegistry` and the `detect`/`status` CLI are retained. `detect_hardware` and `select_device` inspect the local client machine only; they do not verify the remote engine. Local model inference, speech adapters and server hosting are outside this SDK subset. The bundled catalog contains `object-detection` and `mock-detector`; additional declarative tasks can be loaded with `AlgorithmRegistry.load_from_dir` or `OMNISENSE_ALGORITHM_DIR`. Catalog entries and configured engines are not health or quality acceptance.

## HTTP contract

The engine implements `POST /api/algorithms/{algorithm_id}/run`. A single image request is:

```json
{"algorithm_id":"object-detection","input":{"kind":"image_bytes","mime_type":"image/jpeg","data_base64":"BASE64_IMAGE"},"options":{"task":"detect","model_id":"my-detector"}}
```

Multiple images use `input.kind="image_bytes_list"` and `items:[{"mime_type":"image/jpeg","data_base64":"..."}]`. The response must be an object with the matching `algorithm_id`, an `outputs` array and a status from `ok`, `missing_model`, `unknown_algorithm`, `unsupported_input`, `invalid_request`, `runtime_error`, or `timeout`. Unknown response fields are preserved. Output semantics and supported tasks belong to the chosen engine; a successful request is not an accuracy or model-rights claim.

Per-algorithm `OMNISENSE_ENGINE_URL/MODEL/TOKEN/TIMEOUT_SECS__<ALGORITHM_ID>` override global values; non-ASCII/non-alphanumeric ID characters become underscores in the setting suffix. An operator-configured model overrides the caller's alias. URLs and credentials come from process settings, never request options. HTTPS is required for remote hostnames/public endpoints; explicit loopback and private IP addresses may use HTTP. Redirects, environment proxies and automatic retries are disabled. Input and response content are bounded to 32 MiB, image lists to 64, timeout to 1–3600 seconds. Keep private inputs in their authorized region and do not enable request-body logging.

## Sources and rights

The retained generic client, CLI, registry and client-hardware helpers are authored within OmniSense; exact retained-file hashes and source lineage are recorded in `PROVENANCE.json`. This export uses fresh history and contains no historical deployment material. Project code is Apache-2.0 under `LICENSE`. The reviewed runtime installs are [Pillow 13.0.0](https://github.com/python-pillow/Pillow/tree/13.0.0) ([MIT-CMU license](https://github.com/python-pillow/Pillow/blob/13.0.0/LICENSE), including the retained PIL copyright notices) and [PyYAML 6.0.3](https://github.com/yaml/pyyaml/tree/6.0.3) ([MIT license](https://github.com/yaml/pyyaml/blob/6.0.3/LICENSE)). They are installed separately; their code is not copied into this SDK wheel. Upstream wheels may include additional native libraries and notices, which must be retained when redistributing those dependencies. Apache-2.0 covers this SDK's own code only. These exact installs are the reviewed versions, not a claim that every historical lower bound or future release has been audited.

No third-party model implementation, weight or training data is bundled or licensed by this SDK. Engine code, each exact model revision, training data and vendor runtime have independent terms. Conversion, encryption, renaming or an HTTP/process boundary does not grant rights or erase applicable obligations. Operators must verify those sources and obtain any required commercial rights. This snapshot makes no blanket commercial-model or accuracy promise.

## Verification scope

Build the SDK with `python -m build --no-isolation` after explicitly installing the exact build tools recorded in `PROVENANCE.json`. GitHub CI scans the reviewed source and complete new history with Gitleaks, compares every wheel/sdist payload to source, cold-installs and compiles the SDK, checks CLI help, and audits every installed third-party build/runtime dependency with strict pip-audit and no advisory exceptions. The project source itself is reviewed through provenance rather than represented as a dependency from a package index. Each CI artifact records its exact source commit, resolved build/audit dependency lists and SHA256 checksums. Behavioral boundary and engine interoperation acceptance are one-off manual checks using a cold-installed wheel and synthetic or otherwise authorized input; they are distinct from CI build/packaging checks and are not shipped as a test suite. CPU interoperation, CUDA execution and TensorRT build/inference are separate evidence; this SDK source publication is not a production deployment or a formal hardware-certified release.
