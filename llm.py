import json
import ollama

model = 'llama3.1'
#model = 'gemma2:2b'
system_prompt_content = """
    You are a friendly chatbot that will engage the user in conversation
"""
system_prompt = {
    'role': 'system',
    'content': system_prompt_content
}

class Llm:
    def __init__(self):
        self.model = model
        self.system_prompt = system_prompt
        self.messages = []
        # preload the model so it's ready for use
        empty = ollama.chat(self.model, [])

    def chat(self, user_content):
        response = None
        cycle_count = 0
        while response is None:
            response = self.execute_llm(user_content)
            cycle_count += 1

        print(f'cycle count: {cycle_count}')

        user_prompt = {'role':'user', 'content': user_content}
        self.messages.append(user_prompt)
        self.messages.append(response['message'])
        return(response['message']['content'])

    def execute_llm(self, user_content):
        message_payload = []
        message_payload.append(self.system_prompt)
        message_payload += self.messages
        user_prompt = {'role':'user', 'content': user_content}
        message_payload.append(user_prompt)
        response =  ollama.chat(self.model, messages=message_payload, options={'num_ctx': 50000})
        # verify that we got a real response that's usable

        return response


    def test_input(self, prompt):
        answer = self.chat(prompt)
        print('{}: {}'.format(prompt, answer))
        return answer

def main():
    l = Llm()
    l.test_input('Hi there how is your day')
    l.test_input('can you tell me a bed time story?')

if __name__ == '__main__':
    main()
