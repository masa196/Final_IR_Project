# Information Retrieval Systems - Practical Project 2026

**Project Title:** Building an Information Retrieval System  
**Course Instructors:** * **Theory Lecturer:** Dr. Obai Sandouk  
* **Lab Instructors:** Eng. Marwa Al-Daya, Eng. Selema Al-Mhairi  

---

## Project Description
The objective of this project is to build a customized search engine capable of searching and retrieving documents from two different datasets. Information Retrieval (IR) principles must be applied to design and implement a search engine that can handle user queries and return textually relevant results in natural language.

### Dataset Requirements
* Each group must select **two different datasets** from the following link: [ir-datasets.com](https://ir-datasets.com).
* **Condition:** Each dataset must contain more than **200K documents**.
* The selected datasets must include **testing data** and **qrel** (i.e., query examples and the most relevant results mapped to those queries).
* **Restrictions:** The `Antique` dataset is strictly prohibited. One of the proposed datasets can be replaced with an external one, subject to approval from the lab instructor.

---

## Core System Requirements

The implemented Information Retrieval System must include the following components:

### 1. Data Pre-Processing
After downloading the datasets, the data must be processed in a manner the group deems appropriate (e.g., *Stemming, Lemmatization, Normalization, etc.*).

### 2. Document Representation
Documents in each dataset must be represented using all of the following methods:
* **VSM_TF-IDF** representation.
* **Embedding** representation (e.g., *Word2Vec, BERT, etc.*).
* **BM25** representation.
* **Hybrid Representation:** This must be implemented in two ways:
  * **Serial** * **Parallel**

> **Notes on Representations:**
> * When using the hybrid representation in **Parallel**, you must employ result fusion methods (**Fusion Methods**) to calculate the final document scores (**Scoring**).
> * For **BM25**, you must provide a way to visualize coefficient adjustments based on the query during execution via the User Interface, or clearly justify the chosen baseline coefficient values within the project report.
> * The User Interface must provide an option allowing the user to choose between searching via **Serial Hybrid Representation** or **Parallel Hybrid Representation**.
> * Multiple embedding model types can be combined together when using the **Parallel Hybrid** representation.

### 3. Indexing
Build one or more appropriate indexes for each dataset (e.g., *Inverted Index*, etc.) to retrieve documents quickly, efficiently, and with optimal indexing term selection.

### 4. Query Processing
Queries must be processed using the exact same pre-processing techniques and represented using the same document representation methods to ensure strict compatibility between queries and retrieved documents.

### 5. Query Refinement
Apply enhancements to queries to increase result accuracy. This includes techniques such as:
* **Query formulation assistance / Query suggestion**
* **Query boosting** using information from the user's past search history.
* Adding **synonyms** to the query.
* **Query spell correction**, etc.

### 6. Query Matching & Ranking
Build a matching function to compare the query representation against the documents and rank the results based on the highest similarity scores. The matching method appropriate for each representation model must be adopted (e.g., *Cosine Similarity* for VSM/Embeddings).

### 7. Service-Oriented Architecture (SOA)
The system must be designed according to the **Service-Oriented Architecture (SOA)** concept, where the system is divided into a set of independent, decoupled services. Each part must be responsible for a specific task and capable of being operated, tested, and developed separately.

For example, the system can be decoupled into services such as:
* **Preprocessing Service**
* **Indexing Service**
* **Retrieval Service**
* **Ranking & Evaluation Service**
* **Query Refinement Service**
* **Frontend Service or API Gateway**

**SOA Design Considerations:**
* Achieve a clear separation of concerns between services.
* Adopt an appropriate communication protocol between services (e.g., *REST API, Message Queue, RPC, etc.*).
* Write clean, organized, maintainable, and scalable code.
* Ensure the ability to run or test each service independently.
* Apply best practices and appropriate **Design Patterns** to optimize performance and scalability.
* Explain the system architecture and communication mechanisms between services in the report using an **Architecture Diagram**.

*Grading for this section increases based on how organized, flexible, and professional the service design is regarding:*
* **Clean Architecture**
* **Scalability**
* **Maintainability**
* **Loose Coupling**
* **Reusability**
* *It is preferred to justify the choice of architecture and technologies used, explaining how they contributed to improving system performance or easing future development.*

---

## 8. System Evaluation
The performance of the information retrieval system must be evaluated using standard IR metrics to verify the quality and relevance of the retrieved results.

The following metrics must be calculated at a minimum for **each representation model** and for **each dataset**:
* **MAP** (Mean Average Precision)
* **Recall**
* **Precision@10**
* **nDCG** (Normalized Discounted Cumulative Gain)

**Evaluation Scenarios:**
The evaluation must be conducted in the following scenarios:
1. **Before** applying additional features.
2. **After** applying additional features.

The report must provide a clear analysis of the results, highlighting:
* The impact of each representation method on retrieval quality.
* Performance comparison across different models (*TF-IDF, BM25, Embeddings, Hybrid*).
* The extent to which additional features contributed to improving results or retrieval speed.
* Justification for the chosen models and parameters based on practical empirical results.

> **Warning:** Any system that yields exceptionally low or illogical evaluation results relative to the nature of the used dataset will be rejected.

---

## 9. User Interface (UI - Web or Mobile Application)
Build an easy-to-use interface that includes:
* Ability to select the dataset from the interface before executing the query, and accepting user queries.
* Displaying relevant results from the dataset.
* An option to choose between executing **Basic requests only** or executing **Basic + Additional Features**.
* Ability to control probabilistic model parameters (BM25 coefficients) directly through the interface.
* Ability to select the preferred hybrid representation model configuration via the interface.

---

## Additional System Features
Depending on the group size, additional features must be implemented. You may propose a different feature, provided it is approved by the lab instructor or professor.

1. **RAG** (Retrieval-Augmented Generation)
2. **Use Vector Stores**
3. **Multilingual Retrieval System**
4. **Crawling**
5. **Distributed Information Retrieval**
6. **Documents Clustering**
7. **Personalization**
8. **Topic Detection**
9. **Agents**
10. **LTR** (Learning to Rank)

---

## Submission Requirements

The final submission must include:
1. **Detailed Report (in Arabic):** Describing the design and implementation of the information retrieval system, including references/sources.
   * Description of the 2 used datasets.
   * Description of the project steps for each service within the system.
   * Description of the system architecture (**System Architecture Diagram**) showing the services and their communication flows according to **SOA**.
   * Evaluation reports as specified.
   * Work distribution among group members.
2. **Executable Version:** A live, ready-to-query version of both search engines to be tested during the interview.
3. **GitHub Repository Link:** Containing the project's source code along with a clear `README.md` explaining the code structure.

---

## Organizational Notes
* **Programming Language:** **Python** exclusively.
* **Group Size & Requirements:** A group can consist of **5 to 7 members**, scaling as follows:
  * If the group has **5 students**: **1 additional feature** is required.
  * If the group has **6 students**: **2 additional features** are required.
  * If the group has **7 students**: **3 additional features** are required.
* **Final Submission Deadline:** **March 7th (7/3)**.