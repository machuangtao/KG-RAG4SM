import openai
from typing import Tuple, Optional, Union
from transformers import pipeline

class KGRAG_for_Schema_Matching:
    def generate_system_prompt(self) -> str:
        return """
                You are an expert in schema matching and data integration. 
                Your task is to analyze the attribute 1 with its textual description 1 and attribute 2 with its textual description 2 from source and target schema in the given question, and specify if the attribute 1 from source schema is semantically matched with attribute 2 from the target schema.
                In some questions, there is the knowledge graph context that might be helpful for you to answer. In this case, you will need to consider the provided context to make the correct decision. \n\n

                Here are some examples of the schema matching questions with correct answers and explanations that you need to learn before you start to analyze the potential mappings:
                Example 1:
                Attribute 1 death-person_id and its description 1 the death domain contains the clinical event for how and when a person dies. a person can have up to one record if the source system contains evidence about the death; a foreign key identifier to the deceased person. the demographic details of that person are stored in the person table. 
                Attribute 2 beneficiarysummary-desynpuf_id and its description 2 beneficiarysummary pertain to a synthetic medicare beneficiary; beneficiary code. 
                Are attribute 1 and attribute 2 semantically matched with each other?
                Here is the correct answer and the explanations for the above-given example question: 1 
                Explanation: they are semantically matched with each other because both of them are unique identifiers for each person. Even if the death-person_id refers to the unique identifier of the person in the death table and beneficiarysummary-desynpuf_id refers to the unique identifier of the person beneficiary from beneficiarysummary table, they are semantically matched with each other. \n\n
                        
                Example 2:
                Attribute 1 death-person_id and its description 1 the death domain contains the clinical event for how and when a person dies. a person can have up to one record if the source system contains evidence about the death.;a foreign key identifier to the deceased person. the demographic details of that person are stored in the person table. 
                Attribute 2 beneficiarysummary-bene_birth_dt and its description 2 beneficiarysummary pertain to a synthetic medicare beneficiary; date of birth. 
                Are attribute 1 and attribute 2 semantically matched with each other?
                Here is the knowledge graph context that might be helpful for you to answer the above schema matching question: 
                death (Q4), has part(s) of the class (P2670), date of death (Q18748141) -> date of death (Q18748141), opposite of (P461), date of birth (Q2389905) | human (Q5), has characteristic (P1552), age of a person (Q185836) -> age of a person (Q185836), uses (P2283), date of birth (Q2389905)
                Here is the correct answer and the explanations for the above-given example question: 0
                Explanation: they are not semantically matched with each other, because death-person_id is a unique identifier for each person in death table and bene_birth_dt is the date of birth of person in beneficiarysummary table. From the above context, we can found that date of death is opposite of date of birth, they are not semantically matched with each other.\n\n

                Example 3:
                Attribute 1 drug_exposure-stop_reason and its description 1 the 'drug' domain captures records about the utilization of a drug when ingested or otherwise introduced into the body. a drug is a biochemical substance formulated in such a way that when administered to a person it will exert a certain physiological effect. drugs include prescription and over-the-counter medicines, vaccines, and large-molecule biologic therapies. radiological devices ingested or applied locally do not count as drugs.;the reason the drug was stopped. reasons include regimen completed, changed, removed, etc. 
                Attribute 2 medications-reasondescription and its description 2 patient medication data; description of the reason code. 
                Are attribute 1 and attribute 2 semantically matched with each other?
                Here is the knowledge graph context that might be helpful for you to answer the above schema matching question: 
                Drug Exposure (C41362):Contact with drug, isa, Chemical Exposure (C36290) | reason for stopping medication (120234), isa, Medication discontinued(274512008)-->has_associated_procedure, Drug therapy(416608005), has_direct_substance, Drug or medicament(410942007)
                Here is the correct answer and the explanations for the above-given example question: 0
                Explanation: they are not semantically matched with each other, because drug_exposure-stop_reason is the reason for stopping the drug exposure and medications-reasondescription is the description of the reason code. From the above context, even if we can found that there is a connection among the drug exposure stop reason and reason for stopping medication but this connection is a sub-concept relation.  This aligns closely with Attribute 1 and shows that stop reason is a well-defined sub-concept distinct from more general medication-reason descriptors, which is not semantically matched with each other. \n\n
                        
                Remember the following tips when you are analyzing the potential mappings.
                Tips:
                (1) Some letters are extracted from the full names and merged into an abbreviation word.
                (2) Schema information sometimes is also added as the prefix of abbreviation.
                (3) Please consider the abbreviation case. 
                (4) Please consider the knowledge graph context to make the correct decision when it is provided.
                """

    def generate_user_prompt(self, question: str, paths: Optional[str]) -> str:
        prompt = f"""Based on the provided example and the following knowledge graph context, please answer the following schema matching question:
        
        {question}

        Knowledge Graph Context:
        {paths if paths and paths != ['null'] else "No available knowledge graph context, please make the decision yourself. "}

        Please respond with the label: 1 if attribute 1 and attribute 2 are semantically matched with each other, otherwise respond lable: 0.
        Do not mention that there is not enough information to decide.
        """
        return prompt
    
    def get_llm_response(self, system_prompt: str, user_prompt: str, model: Union[str, pipeline]) -> str:
        if isinstance(model, str) and model.startswith('gpt'):
            client = openai.Client()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = client.chat.completions.create(
                model=model,
                messages=messages
            )
            return response.choices[0].message.content
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            terminators = [
                model.tokenizer.eos_token_id,
                model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            
            responses = model(
                messages,
                eos_token_id=terminators,
                max_new_tokens=4096,
                do_sample=True,
                temperature=0.5,
                top_k=1,
                top_p=0.9,
                pad_token_id=model.tokenizer.eos_token_id
            )
            answer = responses[0]['generated_text'][-1]["content"].strip()
            return answer
    
    def kgrag_query_for_schema_matching(self, question: str, paths: Optional[str], model) -> Tuple[str, str, str]:
        system_prompt = self.generate_system_prompt()
        user_prompt = self.generate_user_prompt(question, paths)
        response = self.get_llm_response(system_prompt, user_prompt, model)
        return system_prompt, user_prompt, response