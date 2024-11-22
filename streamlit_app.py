import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Title and description
st.title("💬 Chatbot Using Fine-Tuned GPT-2")
st.write(
    "This chatbot uses a fine-tuned GPT-2 model hosted on Hugging Face to generate responses. "
    "You can interact with the chatbot below."
)

# Load the fine-tuned model and tokenizer
@st.cache_resource
def load_model():
    model_name = "akaasula/gpt2_ft_ai_article"  # Replace with your Hugging Face repo
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input field
if user_input := st.chat_input("Type your message here..."):

    # Display the user's input
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate a response from the fine-tuned model
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            input_ids = tokenizer.encode(user_input, return_tensors="pt")
            input_ids = input_ids.to("cuda" if torch.cuda.is_available() else "cpu")
            model.to("cuda" if torch.cuda.is_available() else "cpu")

            # Generate response
            output_ids = model.generate(
                input_ids,
                max_length=200,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Display the assistant's response
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
