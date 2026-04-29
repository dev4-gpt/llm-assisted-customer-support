import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def add_md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def add_code(code):
    cells.append(nbf.v4.new_code_cell(code))

# ---------------------------------------------------------
# SECTION 1: Setup & Core Configuration
# ---------------------------------------------------------
add_md("""
# Comprehensive End-to-End Showcase: LLM-Augmented Customer Support
**Author:** Aryaman Dev

This notebook demonstrates the entire end-to-end pipeline of the `llm-assist` project.
It touches upon **every major directory and component**:
- **`app/`**: Core API and Business Services (Triage, Quality, Pipeline, RAG)
- **`data/`**: Raw datasets, golden evaluation sets, and policy snippets
- **`scripts/`**: Automation scripts for EDA and Offline Evaluation
- **`evaluation/`**: Splitting strategies and Custom Metrics
- **`tests/`**: Unit and Integration Testing

## Section 1: Project Setup & Core Configuration
First, we resolve our root directory and ensure that our application loads the `.env` file correctly using `app.core.config`.
""")

add_code("""
import sys
import os
from pathlib import Path

# Resolve project root (works if run from notebooks/ or repo root)
ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

print(f"Working Directory: {ROOT}")

# Load the Core Configuration (from app/core/config.py)
from app.core.config import get_settings
from app.core.logging import configure_logging

settings = get_settings()
configure_logging(log_level=settings.app_log_level, json_logs=settings.app_env == "production")

print(f"Active LLM Profile: {settings.llm_profile}")
print(f"LLM Provider: {settings.llm_provider}")
""")

# ---------------------------------------------------------
# SECTION 2: Data Engineering & EDA
# ---------------------------------------------------------
add_md("""
## Section 2: Data Engineering & Exploratory Data Analysis (EDA)
Understanding class imbalance and token length is critical before training any baseline models or designing LLM prompts.
In this section, we load the raw Kaggle dataset, visualize its skewed distribution, and implement an **Undersampling Strategy** to explicitly solve the class imbalance problem.
""")

add_code("""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# Load raw dataset
data_path = ROOT / "data" / "raw" / "tickets_labeled.csv"
if data_path.exists():
    df = pd.read_csv(data_path).dropna(subset=["text", "category"])
    
    # Filter to the top 5 relevant categories to clean the data
    valid_categories = ['general_inquiry', 'billing', 'feature_request', 'authentication', 'technical_bug']
    df = df[df['category'].isin(valid_categories)]
    
    # 1. Plot Class Imbalance
    plt.figure(figsize=(10, 5))
    ax = sns.countplot(data=df, y='category', order=df['category'].value_counts().index, palette="viridis")
    plt.title("Original Class Distribution (Massive Imbalance)")
    plt.xlabel("Number of Tickets")
    plt.ylabel("Category")
    plt.show()
    
    # 2. Text Length Distribution
    df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(10, 5))
    sns.histplot(df['text_length'], bins=50, kde=True, color="coral")
    plt.title("Ticket Word Count Distribution")
    plt.xlabel("Word Count")
    plt.xlim(0, 150)
    plt.show()
    
    # 3. Solving Class Imbalance via Undersampling
    print("\\n--- Solving Class Imbalance ---")
    min_class_size = df['category'].value_counts().min()
    print(f"Minority class size is {min_class_size}. Undersampling all majority classes to match...")
    
    balanced_df = df.groupby('category').sample(n=min_class_size, random_state=42)
    
    print("\\nNew Balanced Distribution:")
    display(balanced_df['category'].value_counts().to_frame())
    
else:
    print("Raw dataset not found. Please ensure data/raw/tickets_labeled.csv is downloaded.")
""")

# ---------------------------------------------------------
# SECTION 3: Baseline Training & Hyperparameter Tuning
# ---------------------------------------------------------
add_md("""
## Section 3: Baseline Training & Hyperparameter Tuning
While large language models (LLMs) provide the reasoning backbone, we also utilize classical Machine Learning (TF-IDF + Logistic Regression) as a fast, cheap **Hybrid Hint** mechanism.

Here, we will perform a **Hyperparameter Search** over the `C` parameter in Logistic Regression, using a subset of our data to find the model that yields the highest accuracy. This model is saved to `artifacts/triage_baseline.joblib`.
""")

add_code("""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the authentic Kaggle dataset
data_path = ROOT / "data" / "raw" / "tickets_labeled.csv"
if not data_path.exists():
    print(f"File not found: {data_path}. Creating a dummy dataset for demonstration.")
    data = pd.DataFrame({
        "text": ["My password is not working", "Charge is too high", "Add dark mode", "Cancel my sub"],
        "category": ["authentication", "billing", "feature_request", "cancellation"]
    })
else:
    data = pd.read_csv(data_path).dropna(subset=["text", "category"])
    
    # Filter to the top 8 most frequent categories to remove noisy edge cases
    top_categories = data['category'].value_counts().nlargest(8).index
    data = data[data['category'].isin(top_categories)]
    
    # Take up to 10,000 rows of authentic data
    data = data.sample(n=min(10000, len(data)), random_state=42)

X = data['text']
y = data['category']

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Unified Hyperparameter Grid Search ---
print("Starting Unified Hyperparameter Grid Search (TF-IDF + LinearSVC)...")

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', sublinear_tf=True)),
    ('clf', LinearSVC(class_weight='balanced', dual=False, max_iter=2000))
])

param_grid = {
    'tfidf__max_features': [1000, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 25000],
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'clf__C': [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
}

# n_jobs=-1 uses all CPU cores to run the grid search in parallel
grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print(f"\\nBest Parameters found: {best_params}")

preds = best_model.predict(X_test)
best_accuracy = accuracy_score(y_test, preds)

print(f"\\nTest Set Accuracy with Best Model: {best_accuracy:.4f}\\n")
print("Classification Report (F1 Scores):")
print(classification_report(y_test, preds))

joblib.dump(best_model, ROOT / "artifacts" / "triage_baseline.joblib")
print("Saved baseline model to artifacts/triage_baseline.joblib")
""")

# ---------------------------------------------------------
# SECTION 4: The LLM Service Layer (RAG & Fallback)
# ---------------------------------------------------------
add_md("""
## Section 4: The LLM Service Layer (RAG & Fallback)
Next, we instantiate our core services from the `app/services/` directory.

- **`RAGService`**: Retrieves policy documents from `data/policy_snippets.json` (lexical or embedding search).
- **`IntentFallbackService`**: If the LLM hallucinates an invalid category (e.g., "refund_request" instead of "billing"), this service uses embeddings to deterministically map it back to the allowed taxonomy.
""")

add_code("""
import asyncio
from app.models.domain import RAGContextRequest
from app.services.rag_service import RAGService
from app.services.intent_fallback_service import IntentFallbackService

# Both services require the 'settings' object initialized in Section 1
# Initialize RAG
rag_service = RAGService(settings)
query = "What happens if I have a duplicate charge?"
request = RAGContextRequest(query=query)
response = rag_service.retrieve(request, top_k=1)

print(f"RAG Retrieval for '{query}':")
if response.snippets:
    print(f" - Found Snippet ID: {response.snippets[0].id}")
else:
    print(" - No snippets found (is policy_snippets.json populated?)")

# Initialize Fallback Service
fallback_service = IntentFallbackService(settings)
invalid_label = "refund"
recovered = fallback_service.map_to_valid_category(invalid_label)
print(f"\\nFallback Recovery:\\n - Original hallucinated label: '{invalid_label}'\\n - Recovered valid taxonomy: '{recovered}'")
""")

# ---------------------------------------------------------
# SECTION 5: End-to-End Pipeline Execution
# ---------------------------------------------------------
add_md("""
## Section 5: End-to-End Pipeline Execution
We will now use `PipelineService` to orchestrate `TriageService` and `QualityService` simultaneously!
We'll load a mock ticket from `data/fixtures/` and execute it through the LLM.
""")

add_code("""
import json
from app.models.domain import PipelineRequest
from app.services.llm_client import LLMClient
from app.services.triage_service import TriageService
from app.services.quality_service import QualityService
from app.services.pipeline_service import PipelineService

# Setup the core services
llm_client = LLMClient(settings)
triage_service = TriageService(llm_client, settings, rag_service)
quality_service = QualityService(llm_client, settings, rag_service)
pipeline_service = PipelineService(triage_service, quality_service, settings)

# Load a fixture (Zendesk Mock Data)
fixture_path = ROOT / "data" / "fixtures" / "zendesk_ticket.json"
with open(fixture_path, "r") as f:
    ticket_data = json.load(f)

print(f"Input Ticket:\\n{ticket_data['ticket']['description']}\\n")

request = PipelineRequest(
    ticket_text=ticket_data['ticket']['description'],
    agent_response="I will look into this for you. Give me 5 days.",
    include_policy_context=True
)

# Run the Pipeline (Synchronous via Threading)
print("Calling LLM Pipeline (Triage + Quality concurrent via ThreadPool)...")
result = pipeline_service.run(request)

print("\\n--- Pipeline Output ---")
print(f"Category: {result.triage.category.value}")
print(f"Priority: {result.triage.priority.value}")
print(f"Quality Passed: {result.workflow_passed}")
print(f"Quality Coaching: {result.quality.coaching_feedback}")
""")

# ---------------------------------------------------------
# SECTION 6: Offline Evaluation on Golden Set
# ---------------------------------------------------------
add_md("""
## Section 6: Offline Evaluation on Golden Set
To prove our LLM works at scale, we use the `scripts/run_offline_eval.py` methodology to evaluate tickets from `data/golden/eval_set.jsonl`.
This invokes our `evaluation/metrics.py` to compute F1 Scores and Confusion Matrices.
""")

add_code("""
import subprocess
import os
import sys
from IPython.display import display, Markdown

# We execute the offline evaluation script in a subprocess.
# By passing EVAL_LLM=1, we tell the script to use the real LLM endpoint instead of mocking.

print("Running Offline Evaluation Script...")
try:
    # Notice how we pass our clean environment to the subprocess
    clean_env = os.environ.copy()
    
    subprocess.check_call(
        [
            sys.executable,
            "scripts/run_offline_eval.py",
            "--data",
            str(ROOT / "data" / "golden" / "eval_set.jsonl")
        ],
        cwd=str(ROOT),
        env=clean_env
    )
    print("\\nEvaluation Script Completed!")
    
    # Display the resulting Markdown summary directly in the notebook!
    summary_path = ROOT / "artifacts" / "eval" / "summary.md"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            display(Markdown(f.read()))
            
except Exception as e:
    print(f"Evaluation failed: {e}")
""")

# ---------------------------------------------------------
# SECTION 7: Quality Assurance & Testing
# ---------------------------------------------------------
add_md("""
## Section 7: Quality Assurance & Testing
Our repository contains a robust suite of Unit and Integration tests in the `tests/` directory.
We can invoke `pytest` to prove that the entire system architecture is stable.
""")

add_code("""
print("Running test suite...")
cp = subprocess.run(
    ["pytest", "tests/", "-q", "--no-cov", "--disable-warnings"],
    cwd=str(ROOT),
    text=True,
    capture_output=True
)

print(cp.stdout)
if cp.returncode == 0:
    print("✅ All tests passed successfully!")
else:
    print("❌ Some tests failed.")
    print(cp.stderr)
""")

nb.cells = cells

output_path = "notebooks/llm_assist_end_to_end_final.ipynb"
with open(output_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Created {output_path}")
