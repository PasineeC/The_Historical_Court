import os
import logging
import google.cloud.logging
from dotenv import load_dotenv

from google.adk import Agent
from google.adk.agents import SequentialAgent, LoopAgent, ParallelAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.langchain_tool import LangchainTool
from google.adk.tools import exit_loop

# Import Wikipedia Tools
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Setup Logging
try:
    cloud_logging_client = google.cloud.logging.Client()
    cloud_logging_client.setup_logging()
except Exception:
    logging.basicConfig(level=logging.INFO)

load_dotenv()
model_name = os.getenv("MODEL", "gemini-1.5-flash") # Default หากลืมตั้งใน .env

# ==========================================
# 1. TOOLS DEFINITION
# ==========================================

def update_research_data(tool_context: ToolContext, field: str, new_info: str) -> dict[str, str]:
    """จัดการข้อมูลใน State โดยรองรับทั้ง String และ List เพื่อป้องกัน Error"""
    current_data = tool_context.state.get(field, "")
    
    # ป้องกันกรณีข้อมูลเป็น None
    if current_data is None: 
        current_data = ""
    
    # หากเป็น list ให้รวมเป็น string ก่อน (เพื่อความง่ายในการอ่านของ Judge)
    if isinstance(current_data, list):
        current_data = "\n".join(current_data)

    updated_data = f"{current_data}\n\n---\n{new_info}"
    tool_context.state[field] = updated_data
    logging.info(f"Updated field: {field}")
    return {"status": "success"}

import re

def write_verdict_file(tool_context: ToolContext, content: str) -> dict[str, str]:
    topic = tool_context.state.get("PROMPT", "unknown_subject")
    
    # ดึงค่าล่าสุดถ้าเป็น list และล้างขีดล่างที่ซ้ำซ้อน
    if isinstance(topic, list): topic = topic[-1]
    
    # 1. เปลี่ยนเว้นวรรคเป็นขีดล่าง 2. ลบอักขระพิเศษ 3. ยุบขีดล่างที่ติดกันหลายตัวเหลือตัวเดียว
    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', topic)
    clean_name = re.sub(r'_+', '_', clean_name).strip('_')

    filename = f"{clean_name}.txt"
    directory = "court_records"
    
    target_path = os.path.join(directory, filename)
    os.makedirs(directory, exist_ok=True)
    
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"status": "success", "file_path": target_path}

# ตั้งค่า Wikipedia ให้ดึงข้อมูลที่กระชับขึ้นเพื่อป้องกัน Token เต็ม
wiki_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=3000)
wiki_tool = LangchainTool(tool=WikipediaQueryRun(api_wrapper=wiki_wrapper))

# ==========================================
# 2. AGENTS DEFINITION
# ==========================================

admirer_agent = Agent(
    name="admirer_agent",
    model=model_name,
    description="Researches positive achievements.",
    instruction="""
    ROLE: You are 'The Admirer'. Focus ONLY on successes and positive legacy.
    SUBJECT: {PROMPT}
    FEEDBACK: {judge_feedback?}

    TASK:
    1. Search Wikipedia for "{PROMPT} achievements" or "{PROMPT} legacy".
    2. Summarize the positive points clearly.
    3. Use 'update_research_data' to save to 'pos_data'.
    """,
    tools=[wiki_tool, update_research_data]
)

critic_agent = Agent(
    name="critic_agent",
    model=model_name,
    description="Researches controversies and failures.",
    instruction="""
    ROLE: You are 'The Critic'. Focus ONLY on controversies and criticisms.
    SUBJECT: {PROMPT}
    FEEDBACK: {judge_feedback?}

    TASK:
    1. Search Wikipedia for "{PROMPT} controversy" or "{PROMPT} criticism".
    2. Summarize the negative points clearly.
    3. Use 'update_research_data' to save to 'neg_data'.
    """,
    tools=[wiki_tool, update_research_data]
)

judge_agent = Agent(
    name="judge_agent",
    model=model_name,
    description="Reviews the balance of evidence.",
    instruction="""
    ROLE: You are 'The Judge'. Evaluate the evidence for {PROMPT}.
    
    POSITIVE EVIDENCE: {pos_data?}
    NEGATIVE EVIDENCE: {neg_data?}

    LOGIC:
    - If either side is empty or too short: 
        1. Write specific search instructions to 'judge_feedback'.
        2. Use 'update_research_data' to save that feedback.
        3. DO NOT call exit_loop.
    - If both sides are balanced and sufficient:
        1. Call 'exit_loop' immediately.
    """,
    tools=[update_research_data, exit_loop]
)

verdict_writer = Agent(
    name="verdict_writer",
    model=model_name,
    instruction="""
    Create a NEUTRAL historical report for {PROMPT}.
    Using:
    - Positives: {pos_data}
    - Negatives: {neg_data}

    Format as a formal court verdict and save using 'write_verdict_file'.
    """,
    tools=[write_verdict_file]
)

# ==========================================
# 3. ARCHITECTURE & EXECUTION
# ==========================================

investigation_team = ParallelAgent(
    name="investigation_team",
    sub_agents=[admirer_agent, critic_agent]
)

trial_loop = LoopAgent(
    name="trial_loop",
    sub_agents=[investigation_team, judge_agent],
    max_iterations=3
)

# ให้สร้าง Tool ใหม่สั้นๆ สำหรับบันทึกค่าแบบ "เขียนทับ" (Overwrite)
def set_state_value(tool_context: ToolContext, field: str, value: str) -> dict[str, str]:
    """เขียนทับค่าใน State (ไม่ใช่การ Append)"""
    tool_context.state[field] = value
    return {"status": "success"}

# ใน inquiry_agent ให้เปลี่ยนไปใช้ set_state_value แทน append_to_state สำหรับ PROMPT
inquiry_agent = Agent(
    name="inquiry_agent",
    # ...
    instruction="""
    Greet the user and ask for the historical subject.
    Use 'set_state_value' to save the subject name to the field 'PROMPT'.
    """,
    tools=[set_state_value] # เปลี่ยนตรงนี้
)

root_agent = SequentialAgent(
    name="historical_court_system",
    sub_agents=[
        inquiry_agent,
        trial_loop,
        verdict_writer
    ]
)

