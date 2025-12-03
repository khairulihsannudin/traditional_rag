# agents/synthesizer_agent.py - Traditional RAG Answer Generation
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from settings import llm

synthesis_prompt = ChatPromptTemplate.from_template("""You are an expert log analysis assistant.
Your task is to answer the user's question based on relevant log data retrieved from the system.

**Original Question:**
{original_question}

**Retrieved Log Context:**
{log_vector_context}

**Instructions:**
1. First, present the retrieved log information clearly
2. Then provide your analysis and answer to the question
3. Reference specific log entries, timestamps, or patterns when relevant
4. If the logs show errors, failures, or anomalies, explain what they mean
5. If the retrieved context doesn't fully answer the question, acknowledge this

**Format your response as:**

## Retrieved Log Information:
[Present the key log entries found]

## Analysis & Answer:
[Your detailed answer to the user's question based on the logs]""")

synthesis_chain = synthesis_prompt | llm | StrOutputParser()
