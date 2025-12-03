from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from settings import llm

class GuardrailsRouterOutput(BaseModel):
    """
    Combined output for guardrails checking and routing decisions
    """
    decision: Literal["relevant", "irrelevant"] = Field(
        description="Checks if the question is relevant to cybersecurity topics including log analysis, threat detection, vulnerability assessment and attack pattern reconstruction."
    )
    datasource: Literal["log_analysis", "cyber_knowledge"] = Field(
        description="Routes relevant questions to 'log_analysis' for log analysis and specific question related to log in the system or to 'cyber_knowledge' for general cybersecurity questions. Only populated if decision is 'relevant'."
    )

guardrails_router_prompt = ChatPromptTemplate.from_messages([
    (
        "system", 
        """
        You are a gatekeeper and router for a Q&A system that queries a knowledge graph.
        Your task is to:
        1. Determine if a question is answerable by the graph (guardrails)
        2. If relevant, route it to the appropriate tool
        
        The graph contains two types of information:
            1. **Log Data:** Detailed records of system events, including specific users, servers, hosts, processes, and session activities (e.g., login successes or failures).
            2. **Cybersecurity Knowledge:** knowledge base for threat intelligence, including CVE, CWE, CAPEC and MITRE ATT&CK.

        **GUARDRAILS RULES:**
        - A question is **relevant** if it asks about security log analysis such as finding suspicious activities in log or identify specific entities in logs such as users, hosts, ip-address, log events, OR general cybersecurity topics.
        - A question is **irrelevant** if it is completely off-topic (e.g., 'what is the weather?', 'tell me a joke').

        **ROUTING RULES (only apply if question is relevant):**
        1. **Prioritize Log Analysis**: If the question related to log analysis and anything about events in log, route it to 'log_analysis'.
        2. **Use Cyber Knowledge for General Queries**: If the question is about general cybersecurity information and threat intelligence, route it to 'cyber_knowledge'.

        Only allow relevant questions to pass with appropriate routing.
        """
    ),
    ("human", "Question: {question}"),
])

guardrails_router_chain = guardrails_router_prompt | llm.with_structured_output(GuardrailsRouterOutput)