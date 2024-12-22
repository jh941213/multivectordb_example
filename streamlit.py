from typing import Annotated, Sequence, TypedDict, Dict, List
import os
import json
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from tools.vectordb_tool import KTDSVectorDBSearchTool
from tools.perplexity_tool import PerplexityQATool  # 새로 추가

load_dotenv()

# 에이전트 상태 정의
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    chat_history: List[Dict[str, str]]

def create_qa_react_agent():
    """KT DS QA ReAct 에이전트 생성"""
    # 기존 KTDS VectorDB 툴
    qa_tool = KTDSVectorDBSearchTool()
    # 추가 Perplexity 툴
    perplexity_tool = PerplexityQATool()
    
    tools = [qa_tool, perplexity_tool]
    tools_by_name = {tool.name: tool for tool in tools}
    
    # 모델 설정
    model = ChatOpenAI(model="gpt-4o-mini")
    # 여러 툴을 모델에 바인딩
    model = model.bind_tools(tools)
    
    def tool_node(state: AgentState) -> Dict:
        """도구 실행을 처리하는 노드"""
        outputs = []
        last_msg = state["messages"][-1]
        
        if hasattr(last_msg, 'tool_calls'):
            for tool_call in last_msg.tool_calls:
                # tool_call["name"] 이 ktds_qa_search 혹은 perplexity_qa_tool 일 경우 해당 툴을 호출
                tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
                outputs.append(
                    ToolMessage(
                        content=tool_result,  # Dict 형태로 들어올 수 있음
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    )
                )
        return {"messages": outputs, "chat_history": state["chat_history"]}

    def call_model(state: AgentState, config: RunnableConfig) -> Dict:
        """LLM 호출 노드"""
        system_prompt = SystemMessage(content="""
        당신은 KT DS QA 어시스턴트입니다.
        
        - 사내 정보가 필요한 질문의 경우 ktds_qa_search(query, category)를 호출하세요.
        - 사내 DB에서 충분한 정보를 얻지 못했거나 검색 결과가 부족하다고 판단되면 perplexity_qa_tool(query)를 호출하세요.
        - 데이터베이스에는 hr, company, finance, product 카테고리가 있습니다.
        - 도구 호출 결과(검색 결과)를 바탕으로 자연스러운 답변을 생성하세요.
        - 검색 결과가 질문과 관련이 없다면 "관련 정보를 찾을 수 없어요" 라고 답변하세요.
        """)

        all_messages = [system_prompt]
        all_messages.extend([
            HumanMessage(content=h["content"]) if h["role"] == "user" 
            else SystemMessage(content=h["content"]) 
            for h in state["chat_history"]
        ])
        all_messages.extend(state["messages"])
        
        response = model.invoke(all_messages, config)
        return {"messages": [response], "chat_history": state["chat_history"]}

    def should_continue(state: AgentState) -> str:
        """다음 단계를 결정하는 노드"""
        last_msg = state["messages"][-1]
        
        # 마지막 메시지에 tool_calls 가 존재하고,
        # 그 안에 ktds_qa_search 또는 perplexity_qa_tool 이 있다면 'continue' 반환
        if hasattr(last_msg, 'tool_calls') and any(
            call["name"] in ["ktds_qa_search", "perplexity_qa_tool"] for call in last_msg.tool_calls
        ):
            return "continue"
        
        return "end"
    
    # 그래프 구성
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

# Streamlit UI 설정
st.set_page_config(
    page_title="KTDS QA",
    layout="centered",
)

# 세션 상태 초기화
if "agent" not in st.session_state:
    st.session_state.agent = create_qa_react_agent()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 채팅 인터페이스
st.title("KTDS QA Assistant")
st.divider()

# 채팅 기록 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 입력창
if prompt := st.chat_input("메시지를 입력하세요!"):
    # 사용자 메시지 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.chat_message("User").write(prompt)
    
    # 답변 생성
    with st.spinner("🤖 KTds QA 에이전트가 답변을 생성하고 있어요..."):
        try:
            # 실제 처리 시작
            response = st.session_state.agent.invoke({
                "messages": [HumanMessage(content=prompt)],
                "chat_history": st.session_state.chat_history
            })
           
            if response and "messages" in response:
                assistant_message = response["messages"][-1]
                assistant_content = assistant_message.content
                
                # 일반 텍스트 응답 처리
                st.session_state.messages.append({"role": "assistant", "content": assistant_content})
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_content})
                
        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")
            print(f"에러 상세: {e}")

    st.rerun()
