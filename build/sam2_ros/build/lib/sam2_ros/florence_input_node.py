import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import torch
import sounddevice as sd
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Function to record audio from the microphone
def record_audio(duration=10, sample_rate=16000):
    print("Recording audio for {} seconds...".format(duration))
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is finished
    print("Recording finished.")
    return np.squeeze(audio)  # Remove single-dimensional entries

# Function for speech recognition using Whisper
def recognize_speech():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float16

    model_id = "openai/whisper-base"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    duration = 10  # Duration of recording in seconds
    sample_rate = 16000  # Whisper expects a sample rate of 16 kHz
    
    # Record the audio
    audio_data = record_audio(duration=duration, sample_rate=sample_rate)
    
    # Convert the audio to the format expected by the pipeline (dict with 'array' and 'sampling_rate')
    sample = {"array": audio_data, "sampling_rate": sample_rate}
    
    # Perform speech recognition
    result = pipe(sample, generate_kwargs={"language": "english"})
    return result["text"]

class FlorenceInputNode(Node):
    def __init__(self):
        super().__init__('florence_input_node')

        # Create two publishers: one for task_prompt and one for text_input
        self.task_prompt_publisher = self.create_publisher(String, 'task_prompt', 10)
        self.text_input_publisher = self.create_publisher(String, 'text_input', 10)

        # Timer for input loop
        self.timer = self.create_timer(1.0, self.publish_input)

    def publish_input(self):
        # Prompt user to select task type
        task_type = input("Enter 1 for CAPTION_TO_PHRASE_GROUNDING or 2 for OD: ")

        # Create task_prompt based on input
        task_prompt_msg = String()
        if task_type == "1":
            task_prompt_msg.data = "<CAPTION_TO_PHRASE_GROUNDING>"
        elif task_type == "2":
            task_prompt_msg.data = "<OD>"
        else:
            print("Invalid input. Please try again.")
            return

        # Now prompt for how to assign text_input
        input_method = input("Enter 1 to type text_input or 2 for speech input: ")
        text_input = ""

        if input_method == "1":
            text_input = input("Enter text_input (e.g., 'find out vase and fruits'): ")
        elif input_method == "2":
            print("Please speak now...")
            text_input = recognize_speech()  # Call speech-to-text function
            print(f"Recognized Text: {text_input}")
        else:
            print("Invalid input. Please try again.")
            return

        # Create ROS2 String messages for task_prompt and text_input
        text_input_msg = String()
        text_input_msg.data = text_input

        # Publish task_prompt and text_input on separate topics
        self.task_prompt_publisher.publish(task_prompt_msg)
        self.text_input_publisher.publish(text_input_msg)

        # Log the published messages
        self.get_logger().info(f"Published task_prompt: {task_prompt_msg.data}")
        self.get_logger().info(f"Published text_input: {text_input_msg.data}")

def main(args=None):
    rclpy.init(args=args)
    florence_input_node = FlorenceInputNode()
    rclpy.spin(florence_input_node)
    florence_input_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


















# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import String

# class FlorenceInputNode(Node):
#     def __init__(self):
#         super().__init__('florence_input_node')

#         # Create two publishers: one for task_prompt and one for text_input
#         self.task_prompt_publisher = self.create_publisher(String, 'task_prompt', 10)
#         self.text_input_publisher = self.create_publisher(String, 'text_input', 10)

#         # Timer for input loop
#         self.timer = self.create_timer(1.0, self.publish_input)

#     def publish_input(self):
#         # Get task_prompt and text_input from user
#         task_prompt = input("Enter task_prompt (e.g., <CAPTION_TO_PHRASE_GROUNDING>): ")
#         text_input = input("Enter text_input (e.g., 'find out vase and fruits'): ")

#         # Create ROS2 String messages
#         task_prompt_msg = String()
#         text_input_msg = String()

#         task_prompt_msg.data = task_prompt
#         text_input_msg.data = text_input

#         # Publish task_prompt and text_input on separate topics
#         self.task_prompt_publisher.publish(task_prompt_msg)
#         self.text_input_publisher.publish(text_input_msg)

#         # Log the published messages
#         self.get_logger().info(f"Published task_prompt: {task_prompt}")
#         self.get_logger().info(f"Published text_input: {text_input}")

# def main(args=None):
#     rclpy.init(args=args)
#     florence_input_node = FlorenceInputNode()
#     rclpy.spin(florence_input_node)
#     florence_input_node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()
