from prompt.base_prompt import BasePrompt

class PMPPrompt(BasePrompt):
    def generate_prompt(self, text: str) -> str:
        pmp_prompt = f"""
        You will be given a piece of text. First, analyze the text to understand the speaker's intent.
        Then, reflect on the following aspects to determine if the statement is sarcastic or not:
        
        1. The Implicature - What is implied in the conversation beyond the literal meaning?
        2. The Presuppositions - What information in the conversation is taken for granted?
        3. The Intent of the Speaker - What do the speaker(s) hope to achieve with their statement?
        4. The Polarity - Does the last sentence have a positive or negative tone?
        5. Pretense - Is there pretense in the speaker’s attitude?
        6. Meaning - What is the difference between the literal and implied meaning of the statement?
        
        Reflect on the above aspects, and decide if the statement is sarcastic. Answer 'yes' or 'no'.
        Text: "{text}"
        """
        return pmp_prompt
