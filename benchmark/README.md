# EarthBench data

The benchmark data are not stored in this Git repository. From the project
root, download the question manifest and scene files:

```bash
mkdir -p benchmark
curl -L -o benchmark/question.json \
  https://raw.githubusercontent.com/opendatalab/Earth-Agent/main/benchmark/question.json

pip install -U "huggingface_hub[cli]"
hf download Sssunset/Earth-Bench \
  --repo-type dataset \
  --local-dir benchmark/data
```

The resulting layout must be:

```text
benchmark/
├── question.json
└── data/
    ├── question1/
    ├── ...
    └── question248/
```

The paths embedded in `question.json` use the `benchmark/data/questionN`
layout. Do not place the downloaded dataset under `benchmark/EarthBench`
unless you also rewrite those paths.

For parallel runs, create an isolated dataset copy and a matching question
manifest:

```bash
cp -rL benchmark/data benchmark/data1
python benchmark/make_question_variant.py --version 1
```

This produces `benchmark/question1.json`, whose data paths point to
`benchmark/data1/`. Large data directories, generated manifests, and tool
outputs are ignored by Git.
