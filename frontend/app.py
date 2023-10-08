import sys
import streamlit as st
import requests
import json
from collections import defaultdict
from utils import img_to_html, create_graph, plotly_plot, plotly_timeline, strip_debug

try: 
    PORT = int(sys.argv[1])
except:
    PORT = 8000

@st.cache_data
def convert_to_json(chat_history):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return json.dumps(chat_history)


def chat_actions(user_key): 
    st.session_state["chat_history"][user_key].append(
        {"role": "user", "content": st.session_state["chat_input"]},
    )


st.markdown(img_to_html('logo.png', 'Xray-Insight'), unsafe_allow_html=True)
st.write("#### Lets dive deeper into that Chest")
mode = st.sidebar.radio(label="Logging mode", options=["info", "debug"])
plot_type = None
show_details = None
PLOT_OPTIONS = ["timeline", "graph"]

if mode == "debug":
    plot_type = st.sidebar.radio(label="Plot Type", options=PLOT_OPTIONS)
    show_details = st.sidebar.checkbox("Show Details")
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = defaultdict(list)


metadata = st.sidebar.text_input("metadata (dict)", value="{}", key="metadata")
user_key = '0'
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()

prompt = st.chat_input("Enter your message",
                        on_submit=chat_actions,
                        args=(user_key,),
                        key="chat_input",
                        )
for i in st.session_state["chat_history"][user_key]:
    with st.chat_message(name=i["role"], avatar='logo.png' if i["role"]=="assistant" else None):
        st.image(bytes_data)
        if 'content' in i:
            st.write(i["content"])
        if 'debug_message' in i:
            if i["type"] == "json":
                st.json(i["debug_message"], expanded=False)
            else:
                st.write(i["debug_message"])

if prompt:
    prompt = "User: " + prompt
    history = st.session_state["chat_history"][user_key]
    if len(history) > 2:
        past = []
        for chat in history:
            if "content" in chat:
                if chat["role"] == "user":
                    past.append("User: " + chat["content"])
                else:
                    past.append("System: " + chat["content"])
        past = "<sep>".join(past)
        prompt = past + "<sep>" + prompt

    with st.spinner("generating..."):
        files = {"file": (uploaded_file.name, uploaded_file, "image/jpeg")}
        result = requests.post(f"http://localhost:{PORT}/predict/",
                                                                files=files,
                                                                #headers = {"Content-Type": "multipart/form-data"}
                                                                ).json()
    if result:
        with st.chat_message(name="ai", avatar="logo.png"):
            output = result["response"]
            if "links" in result["response"]:
                for name, link in result["response"]["links"].items():
                    if link:
                        output = output.replace(name, f"[{name}]({link})")
            print(output)
            formatted_resp = output.replace("System: ", "").replace("User: ", "").replace("<|endoftext|>", "")
            st.write(formatted_resp)
            st.session_state["chat_history"][user_key].append(
                {
                    "role": "assistant",
                    "content": formatted_resp,
                },
            )
            if result["graph"]:
                # Create a directed graph using NetworkX
                st.toast(f'## Total time: {round(result["graph"][-1]["time"], 3)}s')
            
                if plot_type == "graph":
                    G = create_graph(result)
                    fig = plotly_plot(G)
                    st.plotly_chart(fig, use_container_width=False)

                elif plot_type == "timeline":
                    fig = plotly_timeline(result["graph"])
                    st.plotly_chart(fig, use_container_width=False)

                st.session_state["chat_history"][user_key].append(
                    {
                        "role": "assistant",
                        "debug_message": fig,
                        "type": "figure"
                    },
                )
                if show_details:
                    st.write("details:")
                    st.json(result["graph"], expanded=False)
                    st.session_state["chat_history"][user_key].append(
                        {
                            "role": "assistant",
                            "debug_message": result["graph"],
                            "type": "json"
                        },
                    )
if st.sidebar.button("Refresh Chat"):
    del st.session_state["chat_history"][user_key]
    st.rerun()
st.sidebar.download_button("Download Chat History",
                        convert_to_json(strip_debug(st.session_state["chat_history"], user_key)),
                        file_name="chat_history.json")