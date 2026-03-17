# solemne-data-atelier

Here you'll find everything you need to get started with the Ruse of Reuse hackathon, including data access, method development, and evaluation tools.

## Local Installation
#### 1. Installation
```bash
git clone git@github.com:glsch/solemne-data-atelier.git
cd solemne-data-atelier
python -m pip install -e .
```

#### 2. Download the data
To download all the data delivered with the task, run:
```bash
python -m solemne_data_atelier download
```

##  Google Colab Installation
#### 1. Import the Notebook
* Go to the [repository link] and download the `notebooks/method_evaluation_two_methods.ipynb` file to your computer.
* Open [Google Colab](https://colab.research.google.com/).
* From the welcome dialog, select the **Upload** tab.
* Drag and drop the `method_evaluation_two_methods.ipynb` file into the upload area.

#### 2. Enable GPU Acceleration (CUDA)
To speed up model execution, you should enable a GPU instance:
  * In the top menu bar, click on Runtime.
  * Select Change runtime type.
  * Under the Hardware accelerator dropdown, select a GPU (e.g., T4 GPU).
  * Click Save to apply the changes.

#### 3. Configure Environment Secrets
The notebook requires access tokens to download the necessary models and datasets.
* Click the **🔑 Secrets** icon in the left sidebar of your Colab notebook.
* Click **Add new secret** to create the following two credentials:
  * **`HF_TOKEN`**: Your Hugging Face access token. [Guide on creating one here](https://huggingface.co/docs/hub/en/security-tokens).
  * **`GITHUB_TOKEN`**: Your GitHub personal access token. [Guide on creating one here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).
* Make sure to toggle the **Notebook access** switch on for both secrets so the code can read them.

#### 4. Initialize and Run
* Locate the initialization cell (marked with the **▶️ emoji** for Data Environment Preparation) and click its play button to run it. This will authenticate your session and download all required dependencies.
* Once the setup is complete, navigate to the top menu and select **Runtime > Run all** to execute the rest of the notebook.
* Verify that all cells execute successfully. Once confirmed, you are ready to start tweaking the code!

## Data Structure
After `python -m solemne_data_atelier download`, data is organized under `data/`:

```text
ruse_of_reuse/
├── data/                                # Should be downloaded
│   ├── raw/
│   ├── task/
│   ├── vectorstores/
│   ├── book_mapping.tsv
│   ├── reference_mapping.json
│   └── bible.tsv
├── notebooks/
│   ├── method_evaluation.ipynb          # This notebook
│   └── ...
└── src/                                 # Package code
    ├── ruse_of_reuse/
    └── ...
```
`data/task` is the main folder for method development and evaluation.

## [Optional] Reproducing data preprocessing
Everything that is necessary for running baselines is already included into the downloaded data. 
However, you can task data from raw XML files:
```bash
python -m solemne_data_atelier preprocess
```
This will create `data/task/` from `data/raw/`.  
`validation_preview.html` can be used for quick manual checks of extracted spans.

### [Optional] Re-building vectore stores
The task data is delivered with a vector store containing three collections each providing vectors for all verses of the Bible by three models:
* bowphs/LaBerta
* text-embedding-3-large
* comma-project/modernbert-sentembeddings

However, you can re-create them or produce more collections using some other models.
```bash
python -m solemne_data_atelier vectorstore \
  --hf-model some-huggingface-model \
  --openai-model text-embedding-3-small
```
This will create two additional collections with Bible verse embeddings by `some-huggingface-model` and `text-embedding-3-small`.

## Tweaking the baseline
All tunable parameters are in the final two cells of `notebooks/method_evaluation_two_methods.ipynb`.

### 1. Model / Provider Selection (`build_embedding_method_context`)

| Parameter | Options | Effect |
|-----------|---------|--------|
| `provider` | `"hf"`, `"openai"` | Embedding backend |
| `model_name` | e.g. `"bowphs/LaBerta"`, `"text-embedding-3-large"` | Must match a precomputed Chroma collection |

Available precomputed collections of biblical verse embeddings:
* bowphs/LaBerta
* text-embedding-3-large
* comma-project/modernbert-sentembeddings

### 2. Chunking & Retrieval (`participant_method` → `simple_embedding_method`)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `mode` | `"sentence"` | How problem text is split: `"sentence"`, `"sentence_window"`, or `"char"` |
| `sentences_per_chunk` | `2` | *(sentence_window only)* sentences per sliding window |
| `sentence_stride` | `1` | *(sentence_window only)* step between windows — lower = more overlap |
| `char_chunk_size` | `500` | *(char only)* characters per chunk |
| `char_chunk_overlap` | `100` | *(char only)* overlap between char chunks |
| `min_chunk_chars` | `30` | Chunks shorter than this are dropped |
| `top_k` | `5` | Bible verses retrieved per chunk |
| `similarity_threshold` | `0.825` | Minimum cosine similarity to accept a verse — **most impactful parameter** |


* `similarity_threshold` — lower it to improve recall, raise it for precision
* `top_k` — more candidates per chunk increases recall at the cost of precision
* `mode` — `"sentence_window"` often outperforms single sentences for longer allusions 
* Model — swap to a multilingual or domain-specific model if the language warrants it

### 3. Fast dev runs
Use `max_problems=5` in `run_method_on_dataset` while tuning, then remove the limit for final scoring.