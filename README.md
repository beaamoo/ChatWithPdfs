#Chat With Pdfs

#### Chat with pdfs app using LangChain and OpenAI models 

### Part I: Preparation of the source data that will be used to answer questions (a supporting context for GPT model) 
- Load, convert, and split PDF files into pages using the pypdf library.
- Chunk each page into overlapping sections of a predefined length using LangChain.
- Transform each section into an embedding using the 'Ada' GPT model from OpenAI.
- Store the embeddings in a vector store for similarity querying.
- Utilize an in-memory vector store called FAISS for storing the embeddings.

### Part II: Answering questions with data prepared in Part I
- Transform the user question into an embedding using the 'Ada' OpenAI GPT model, similar to Part 1.
- Conduct a similarity search using the FAISS index to retrieve relevant PDF sections that augment the GPT prompt with contextual information.
- Answer the user's actual question within the context of the retrieved PDF sections.
- Store the chat history to support follow-up user questions, utilizing LangChain's 'memory' feature for this purpose.

