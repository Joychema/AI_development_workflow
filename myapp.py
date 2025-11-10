import streamlit as st 
from transformers import pipeline 

# streamlit page setup 
st.set_page_config(
    page_title="Jojo's chatbot App",
    page_icon="😊",
)

# load pre-trained sentiment analysis model
def load_text_generator():
    text_generator = pipeline("text-generation", model="gpt2")
    text_generator.tokenizer.pad_token = text_generator.tokenizer.eos_token
    return text_generator

# system prompt for the chatbot
SYSTEM_PROMPT = ("""
You are Jojo, a friendly and helpful AI chatbot. Engage in natural and informative conversations with users.
You can assist with answering questions, providing recommendations, and more.
Always respond in a polite and respectful manner.
""" 
)

# conversation prompt
def build_conversation_prompt(user_input, chat_history):
    prompt = SYSTEM_PROMPT + "\n\n"
    for turn in chat_history:
        prompt += f"User: {turn['0']}\nJojo: {turn['1']}\n"
    prompt += f"User: {user_input}\nJojo: "
    return prompt


# page header
st.title("Jojo's Chatbot App 🤖")


# build sidebar
with st.sidebar:
    st.title("About")
    max_new_tokens = st.slider("Max New Tokens", min_value=50, max_value=500, value=150, step=25)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

    if st.button("Clear chat"):
        st.session_state['chat_history'] = []

#  initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

# display chat history
for user_message, bot_response in st.session_state.chat_history:
    # st.chat_message("user").markdown(user_message)
    st.markdown(f"**User:** {user_message}")
    st.markdown(f"**Jojo:** {bot_response}")


# user input
user_input = st.chat_input("Type your message here...")
if user_input:
    # display user message
    st.markdown(f"**User:** {user_input}")

    #  spinner while generating response
    with st.spinner("Jojo is typing ..."):
        # load model
        model = load_text_generator()
        # build prompt
        prompt = build_conversation_prompt(user_input, st.session_state.chat_history)

        # generate response
        response = model(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            pad_token_id=model.tokenizer.eos_token_id,
        )[0]['generated_text']

        # extract bot response
        bot_response = response.split("Jojo:")[-1].strip()
        if "Question:" in bot_response:
            bot_response = bot_response.split("Question:")[0].strip()
        # display bot response
        st.markdown(f"**Jojo:** {bot_response}")
        # update chat history
        st.session_state.chat_history.append((user_input, bot_response))
