# 🚗 Système RAG - Analyse de Documents d'Assurance Automobile en Tunisie
![image1](https://github.com/user-attachments/assets/9049323f-f7ff-4a00-9e03-0281ec25af25)
![image1](https://github.com/user-attachments/assets/c54d2dde-f99c-4bd3-a91b-a785041e989e)

Un système de Retrieval-Augmented Generation (RAG) avancé pour l'analyse intelligente de documents d'assurance avec recherche hybride et génération de réponses par IA.

## 📋 Table des matières

- [Aperçu du projet](#aperçu-du-projet)
- [Architecture et techniques](#architecture-et-techniques)
- [Technologies utilisées](#technologies-utilisées)
- [Pipeline de traitement](#pipeline-de-traitement)

---

## 🎯 Aperçu du projet

Ce projet implémente un système RAG complet pour l'analyse de rapports d'assurance automobile en Tunisie. Il combine plusieurs techniques avancées de NLP et d'IA pour :
- Extraire et structurer le contenu de documents PDF
- Effectuer des recherches sémantiques et par mots-clés
- Générer des réponses contextuelles basées sur les documents
- Évaluer la qualité des réponses générées

---

## 🏗️ Architecture et techniques

### 1. **Extraction et Prétraitement de Documents**

#### Techniques utilisées :
- **PyPDF2** : Extraction de texte depuis PDF
- **Parsing structuré** : Détection automatique de sections, tableaux, et recommandations via regex
- **Normalisation Unicode** : Nettoyage et standardisation du texte (NFKC)
- **Chunking intelligent** : Découpage avec overlap (1000 tokens, 200 tokens de chevauchement)

#### Patterns de détection :
```python
- Marqueurs de pages : r'---\s*PAGE\s+(\d+)\s*---'
- Parties principales : r'^\d+\s*(?:ère|e)\s*PARTIE\s*:\s*(.+?)$'
- Sections hiérarchiques : r'^([IVX]+)\.\s+(.+?)$', r'^([A-Z])\.\s+(.+?)$'
- Recommandations : r'Recommandation\s*N°?\s*(\d+)\s*:\s*(.+?)$'
- Tableaux : r'Tableau\s+(\d+)\.\s*(.+?)$'
```

---

### 2. **Système de Recherche Hybride**

#### A. Recherche par mots-clés (BM25)

**Technique** : BM25 (Best Matching 25) - Okapi
- Algorithme de ranking probabiliste
- Mesure TF-IDF améliorée avec saturation
- Tokenisation simple avec nettoyage de ponctuation

**Paramètres** :
```python
- Suppression des mots < 3 caractères
- Normalisation en minuscules
- Retrait de la ponctuation
```

#### B. Recherche sémantique (Vector Search)

**Modèle d'embeddings** : `paraphrase-multilingual-mpnet-base-v2`
- Architecture : MPNet (Masked and Permuted Pre-training)
- Dimension : 768
- Support multilingue (français inclus)
- Normalisation L2 des vecteurs

**Méthode de similarité** : Cosine Similarity
```
similarity = dot(query_vec, doc_vec) / (||query_vec|| * ||doc_vec||)
```

#### C. Fusion hybride

**Stratégie de pondération** :
```python
score_hybride = (0.4 × score_BM25_normalisé) + (0.6 × score_sémantique_normalisé)
```

**Normalisation** : Min-Max scaling
```python
score_normalisé = score / max(tous_les_scores)
```

**Re-ranking contextuel** :
- Bonus pour présence de recommandations (+0.1)
- Bonus pour présence de chiffres dans les questions numériques (+0.1)
- Bonus pour sections principales (+0.05)

---

### 3. **Génération de Réponses (LLM)**

#### Modèle : Groq API - `llama-3.1-8b-instant`

**Architecture** : LLaMA 3.1 (8 milliards de paramètres)
- Optimisé pour vitesse (inference < 1s)
- Support du français
- Context window : 8192 tokens

**Stratégie de prompting** :
```
Structure :
1. Instructions système (rôle d'expert)
2. Contexte documentaire (top-5 chunks, max 3500 chars)
3. Question utilisateur
4. Consignes de réponse (précision, citations, structure)
```

**Paramètres de génération** :
```python
temperature = 0.3  # Faible pour réponses factuelles
max_tokens = 800    # Réponses concises
```

---

### 4. **Système d'Évaluation**

#### A. Métriques de Récupération

**Cosine Similarity** : Évalue la pertinence des documents récupérés
```python
cos_sim = Σ(query_vec · doc_vec) / n_docs
```

#### B. Métriques de Génération

**BLEU (Bilingual Evaluation Understudy)** :
- Mesure la précision des n-grams (1-4 grams)
- Smoothing : Méthode 1 (add-one smoothing)
```python
BLEU = BP × exp(Σ wₙ log pₙ)
```

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** :
- ROUGE-1 : Unigrams recall
- ROUGE-2 : Bigrams recall  
- ROUGE-L : Longest Common Subsequence
```python
ROUGE-N = Σ n-grams_match / Σ n-grams_référence
```

---

### 5. **Optimisations de Performance**

#### A. Cache de requêtes
- **Méthode** : Hash MD5 des requêtes normalisées
- **Politique** : LRU (Least Recently Used)
- **Taille** : 100 requêtes en mémoire
- **Gain** : ~10x plus rapide pour requêtes répétées

#### B. Indexation efficace
- **BM25** : Tokenisation pré-calculée (pickle)
- **Vector** : Embeddings pré-calculés (NumPy arrays)
- **Option FAISS** : Pour grandes bases (>1000 docs)

---

## 🛠️ Technologies utilisées

### Bibliothèques principales
```python
# Extraction de documents
PyPDF2==3.0.1              # Lecture PDF

# NLP et embeddings
sentence-transformers==2.2.2  # Embeddings multilingues
transformers==4.30.0          # Backend pour sentence-transformers

# Recherche
rank-bm25==0.2.2           # Algorithme BM25
faiss-cpu==1.7.4           # Indexation vectorielle (optionnel)
numpy==1.24.3              # Calculs vectoriels

# Génération
openai==1.3.0              # Client API (Groq compatible)

# Évaluation
nltk==3.8.1                # Tokenisation pour BLEU
rouge-score==0.1.2         # Métriques ROUGE

# Interface
gradio==4.0.0              # Interface web interactive

# Utilitaires
python>=3.8
unicodedata                # Normalisation de texte (stdlib)
```

---

## 📊 Pipeline de traitement

### Phase 1 : Extraction
```
PDF → PyPDF2 → Texte brut → Parsing structuré → Sections + Metadata
                    ↓
            [Cleaning Unicode]
                    ↓
            [Détection patterns]
                    ↓
        Chunking avec overlap (1000/200)
                    ↓
                JSON structuré
```

### Phase 2 : Indexation
```
Chunks → Sentence Transformer → Embeddings (768d)
  ↓                                    ↓
BM25 Tokenization              Normalisation L2
  ↓                                    ↓
Index BM25 (Okapi)            Vector Index (NumPy/FAISS)
```

### Phase 3 : Requête
```
Question utilisateur
    ↓
    ├─→ BM25 Search (40%) ─┐
    └─→ Semantic Search (60%) ─→ Fusion → Top-K chunks
                                     ↓
                              [Re-ranking contextuel]
                                     ↓
                            Contexte + Prompt → Groq LLM
                                     ↓
                            Réponse générée + Sources
```

### Phase 4 : Évaluation
```
Réponse générée + Référence
    ↓
    ├─→ Cosine Similarity (récupération)
    ├─→ BLEU (génération)
    └─→ ROUGE-1/2/L (génération)
    ↓
Métriques d'évaluation
```
