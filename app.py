import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline
import torch
import time # For simulating agent work

# --- Configuration & Model Loading (Placeholder) ---
# You'll need to load your models here. Consider caching them using st.cache_resource.

@st.cache_resource
def load_llm_model():
    # Replace with your actual Mistral 7B model loading
    # model_name = "mistralai/Mistral-7B-Instruct-v0.1" # Example
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # print("LLM Model Loaded")
    # return tokenizer, model
    print("LLM Model Placeholder: Load your language model here.")
    return None, None # Placeholder

@st.cache_resource
def load_diffusion_model():
    # Replace with your actual Stable Diffusion v1.5 model loading
    # model_id = "runwayml/stable-diffusion-v1-5"
    # pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    # if torch.cuda.is_available():
    #     pipe = pipe.to("cuda")
    # print("Diffusion Model Loaded")
    # return pipe
    print("Diffusion Model Placeholder: Load your Stable Diffusion model here.")
    return None # Placeholder

llm_tokenizer, llm_model = load_llm_model()
diffusion_pipe = load_diffusion_model()

# --- Agent Logic (Placeholders) ---

def product_manager_agent(project_idea, chat_history):
    # Simulate PM work
    chat_history.append({"role": "User (to PM)", "content": project_idea})
    st.info("Product Manager Agent is thinking...")
    time.sleep(2) # Simulate processing
    tasks = [
        "Task 1: Define core features based on idea.",
        "Task 2: Identify target audience.",
        "Task 3: Create user stories."
    ]
    response = f"Okay, I've broken down the project idea '{project_idea}' into the following tasks:\n" + "\n".join(tasks)
    chat_history.append({"role": "PM Agent", "content": response})
    return tasks, response

def developer_agent(tasks, chat_history):
    # Simulate Developer work
    tasks_str = "\n".join(tasks)
    chat_history.append({"role": "System (to Developer)", "content": f"Tasks for development:\n{tasks_str}"})
    st.info("Developer Agent is thinking...")
    time.sleep(3) # Simulate processing
    code_snippets = {
        "Task 1": "def feature_one():\n  # TODO: Implement feature one\n  print('Feature one implemented')",
        "Task 2": "class User:\n  def __init__(self, name):\n    self.name = name",
        "Task 3": "# User Story 1: As a user, I want to..."
    }
    response = "I've generated some initial code snippets and thoughts for the tasks:\n"
    for task, code in code_snippets.items():
        response += f"\nFor '{task}':\n```python\n{code}\n```\n"
    chat_history.append({"role": "Developer Agent", "content": response})
    return code_snippets, response

def designer_agent(project_requirements, chat_history):
    # Simulate Designer work
    chat_history.append({"role": "System (to Designer)", "content": f"Project requirements for design:\n{project_requirements}"})
    st.info("Designer Agent is thinking...")
    time.sleep(2) # Simulate processing
    ui_description = "The UI should be clean, modern, with a sidebar for navigation and a main content area. Key colors: #4A90E2 (Blue), #FFFFFF (White)."
    image_prompt = f"A clean and modern user interface for an app about '{project_requirements}', main colors blue and white, with a sidebar and main content area, digital art"
    chat_history.append({"role": "Designer Agent", "content": f"UI Description: {ui_description}\nGenerating image..."})

    image = None
    if diffusion_pipe:
        st.info("Generating UI image with Stable Diffusion...")
        try:
            # Ensure prompt is not empty and is a string
            if image_prompt and isinstance(image_prompt, str):
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast(): # Mixed precision for speed and memory
                        image = diffusion_pipe(image_prompt).images[0]
                else:
                    image = diffusion_pipe(image_prompt).images[0]
                st.success("Image generated!")
            else:
                st.error("Image prompt is invalid for Stable Diffusion.")
        except Exception as e:
            st.error(f"Error generating image: {e}")
            chat_history.append({"role": "Designer Agent", "content": f"Error generating image: {e}"})
    else:
        st.warning("Stable Diffusion model not loaded. Skipping image generation.")
        # Placeholder image if Stable Diffusion is not available
        # image = "https://via.placeholder.com/512x512.png?text=Placeholder+UI+Image"
        chat_history.append({"role": "Designer Agent", "content": "Skipped image generation (model not loaded)."})

    return ui_description, image, image_prompt

# --- Initialize Session State ---
if "pm_chat_history" not in st.session_state:
    st.session_state.pm_chat_history = []
if "dev_chat_history" not in st.session_state:
    st.session_state.dev_chat_history = []
if "design_chat_history" not in st.session_state:
    st.session_state.design_chat_history = []
if "project_idea_input" not in st.session_state:
    st.session_state.project_idea_input = ""
if "tasks_from_pm" not in st.session_state:
    st.session_state.tasks_from_pm = []
if "ui_description_from_designer" not in st.session_state:
    st.session_state.ui_description_from_designer = ""
if "generated_image" not in st.session_state:
    st.session_state.generated_image = None
if "generated_image_prompt" not in st.session_state:
    st.session_state.generated_image_prompt = ""


# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Auto-Agent Network AI App")

st.title("🤖 Auto-Agent Network AI App")
st.caption("Collaborative AI team for project breakdown, coding, and UI design.")

# --- Sidebar for User Input ---
with st.sidebar:
    st.header("Project Idea")
    st.session_state.project_idea_input = st.text_area(
        "Enter your project idea here:",
        st.session_state.project_idea_input,
        height=100
    )

    st.header("Controls")
    run_full_pipeline_button = st.button("🚀 Run Full Automatic Pipeline")
    clear_session_button = st.button("🧹 Clear Session & Chats")

    if clear_session_button:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.markdown("---")
    st.subheader("Models Status:")
    st.caption(f"LLM: {'Loaded' if llm_model else 'Not Loaded (Placeholder)'}")
    st.caption(f"Diffusion: {'Loaded' if diffusion_pipe else 'Not Loaded (Placeholder)'}")
    st.markdown("---")
    st.info("Future Improvements:\n- Real-time multi-agent communication\n- Vector DB for long-term memory\n- Fine-grained user control\n- Richer UI visuals")


# --- Main App Layout (Tabs for Agents) ---
tab_pm, tab_dev, tab_design, tab_pipeline_output, tab_export = st.tabs([
    "🗣️ Product Manager",
    "💻 Developer",
    "🎨 Designer",
    "📋 Full Pipeline Output",
    "📥 Export Chats"
])

# --- Product Manager Tab ---
with tab_pm:
    st.header("Product Manager Agent")
    pm_input = st.text_input("Chat with PM (or use project idea from sidebar):", key="pm_chat_input")
    if st.button("Send to PM", key="pm_send"):
        if pm_input:
            st.session_state.tasks_from_pm, pm_response = product_manager_agent(pm_input, st.session_state.pm_chat_history)
        elif st.session_state.project_idea_input:
            st.session_state.tasks_from_pm, pm_response = product_manager_agent(st.session_state.project_idea_input, st.session_state.pm_chat_history)
        else:
            st.warning("Please enter a project idea or chat message.")

    st.subheader("PM Chat History")
    for chat in st.session_state.pm_chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

# --- Developer Tab ---
with tab_dev:
    st.header("Developer Agent")
    if not st.session_state.tasks_from_pm:
        st.info("Waiting for tasks from the Product Manager. Run PM first or the full pipeline.")
    else:
        st.write("Tasks from PM:", st.session_state.tasks_from_pm)

    dev_input = st.text_input("Chat with Developer (provide context if needed):", key="dev_chat_input")
    if st.button("Send to Developer", key="dev_send"):
        if st.session_state.tasks_from_pm or dev_input:
            # This is a simplified interaction; you might want more sophisticated context passing
            input_for_dev = dev_input if dev_input else "Proceed with tasks."
            if st.session_state.tasks_from_pm and not dev_input:
                _, dev_response = developer_agent(st.session_state.tasks_from_pm, st.session_state.dev_chat_history)
            else: # If there's direct chat or no tasks, pass the input
                _, dev_response = developer_agent([input_for_dev], st.session_state.dev_chat_history)

        else:
            st.warning("Please provide tasks from PM or a direct message.")

    st.subheader("Developer Chat History")
    for chat in st.session_state.dev_chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

# --- Designer Tab ---
with tab_design:
    st.header("Designer Agent")
    if not st.session_state.project_idea_input and not st.session_state.tasks_from_pm:
        st.info("Waiting for project idea or requirements. Run PM/Full Pipeline or provide input.")

    design_input = st.text_input("Provide requirements for the Designer (or it will use project idea):", key="design_chat_input")
    if st.button("Send to Designer", key="design_send"):
        requirements_for_design = design_input if design_input else st.session_state.project_idea_input
        if requirements_for_design:
            st.session_state.ui_description_from_designer, st.session_state.generated_image, st.session_state.generated_image_prompt = designer_agent(
                requirements_for_design, st.session_state.design_chat_history
            )
        else:
            st.warning("Please provide project idea or specific design requirements.")

    st.subheader("Designer Chat History & Output")
    for chat in st.session_state.design_chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    if st.session_state.ui_description_from_designer:
        st.markdown("**UI Description:**")
        st.markdown(st.session_state.ui_description_from_designer)
    if st.session_state.generated_image:
        st.markdown("**Generated UI Image:**")
        st.image(st.session_state.generated_image, caption=f"Generated based on: '{st.session_state.generated_image_prompt}'")


# --- Full Pipeline Output Tab ---
with tab_pipeline_output:
    st.header("Full Pipeline Output")
    if run_full_pipeline_button:
        if not st.session_state.project_idea_input:
            st.error("Please enter a project idea in the sidebar first!")
        else:
            st.session_state.pm_chat_history = []
            st.session_state.dev_chat_history = []
            st.session_state.design_chat_history = []
            st.session_state.generated_image = None
            st.session_state.ui_description_from_designer = ""

            st.subheader("1. Product Manager Agent Processing...")
            with st.spinner("PM is working..."):
                tasks, _ = product_manager_agent(st.session_state.project_idea_input, st.session_state.pm_chat_history)
                st.session_state.tasks_from_pm = tasks
            st.success("PM Agent Finished!")
            st.write("Tasks defined:", st.session_state.tasks_from_pm)
            st.markdown("---")

            st.subheader("2. Developer Agent Processing...")
            with st.spinner("Developer is working..."):
                _, _ = developer_agent(st.session_state.tasks_from_pm, st.session_state.dev_chat_history)
            st.success("Developer Agent Finished!")
            # Display dev output from chat history
            for chat in st.session_state.dev_chat_history:
                if chat["role"] == "Developer Agent":
                    st.markdown(chat["content"])
            st.markdown("---")

            st.subheader("3. Designer Agent Processing...")
            with st.spinner("Designer is working..."):
                ui_desc, img, img_prompt = designer_agent(st.session_state.project_idea_input, st.session_state.design_chat_history) # Or pass tasks/requirements
                st.session_state.ui_description_from_designer = ui_desc
                st.session_state.generated_image = img
                st.session_state.generated_image_prompt = img_prompt
            st.success("Designer Agent Finished!")
            if st.session_state.ui_description_from_designer:
                st.markdown("**UI Description:**")
                st.markdown(st.session_state.ui_description_from_designer)
            if st.session_state.generated_image:
                st.markdown("**Generated UI Image:**")
                st.image(st.session_state.generated_image, caption=f"Generated based on: '{st.session_state.generated_image_prompt}'")

            st.balloons()
            st.success("Full Pipeline Executed Successfully!")

    else:
        st.info("Click 'Run Full Automatic Pipeline' in the sidebar to start.")

    st.subheader("Pipeline Output Summary (if run):")
    if st.session_state.tasks_from_pm:
        st.write("**PM Tasks:**", st.session_state.tasks_from_pm)
    if any(chat["role"] == "Developer Agent" for chat in st.session_state.dev_chat_history):
        st.write("**Developer Output:** See Developer chat history.")
    if st.session_state.ui_description_from_designer:
        st.write("**Designer UI Description:**", st.session_state.ui_description_from_designer)
    if st.session_state.generated_image:
        st.image(st.session_state.generated_image, caption="Latest Generated UI Image")


# --- Export Tab ---
with tab_export:
    st.header("Export Chat Histories")

    def format_chat_history(history, title):
        formatted_text = f"--- {title} Chat History ---\n\n"
        for chat in history:
            formatted_text += f"{chat['role']}:\n{chat['content']}\n\n"
        return formatted_text

    if st.session_state.pm_chat_history:
        st.subheader("Product Manager Chat")
        pm_export_data = format_chat_history(st.session_state.pm_chat_history, "Product Manager")
        st.download_button("Download PM Chat", pm_export_data, "pm_chat_history.txt", "text/plain")
        with st.expander("View PM Chat"):
            st.text(pm_export_data)

    if st.session_state.dev_chat_history:
        st.subheader("Developer Chat")
        dev_export_data = format_chat_history(st.session_state.dev_chat_history, "Developer")
        st.download_button("Download Developer Chat", dev_export_data, "dev_chat_history.txt", "text/plain")
        with st.expander("View Developer Chat"):
            st.text(dev_export_data)

    if st.session_state.design_chat_history:
        st.subheader("Designer Chat")
        design_export_data = format_chat_history(st.session_state.design_chat_history, "Designer")
        st.download_button("Download Designer Chat", design_export_data, "design_chat_history.txt", "text/plain")
        with st.expander("View Designer Chat"):
            st.text(design_export_data)

    if not st.session_state.pm_chat_history and not st.session_state.dev_chat_history and not st.session_state.design_chat_history:
        st.info("No chat histories available to export yet. Run the agents or pipeline.")