import os
import json
import logging
from typing import Dict
# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2", cache_dir="dags/.cache")

# _tokenizer = None
_pdf_converter = None

# def get_tokenizer():
#     """
#     Get the tokenizer for the BERT model.
    
#     Returns:
#         tokenizer: The tokenizer for the BERT model.
#     """
#     global _tokenizer
#     if _tokenizer is None:
#         from transformers import AutoTokenizer
#         logging.info("Loading tokenizer deepset/bert-base-cased-squad2...")
#         _tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2", cache_dir="dags/.cache")
#         logging.info("Tokenizer loaded.")    
    
#     return _tokenizer


def get_pdf_converter():
    """
    Get the PDF converter.
    
    Returns:
        pdf_converter: The PDF converter.
    """
    global _pdf_converter
    if _pdf_converter is None:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        _pdf_converter = PdfConverter(
            artifact_dict=create_model_dict(),
        )
    return _pdf_converter
        

class Data_Processing:
    def __init__(self):
        try:
            self.config_data = json.load(open("dags/config.json", "r"))
            self.data_context = json.load(open("dags/data/data_context.json", "r"))
            self.uploaded_files = self.config_data.get("uploaded_files", [])
            self.file_list = self.config_data.get("file_list", [])
        except Exception as e:
            print(f"Error loading config or data context: {e}")
            self.config_data = {}
            self.data_context = {}
            self.uploaded_files = []
            self.file_list = []

    @staticmethod
    def extract_qa_pairs(raw_data: Dict[str, str]) -> Dict[str, str]:
        """
        Extract question-answer pairs from SQuAD formatted data.
        
        Args:
            raw_data: SQuAD formatted data dictionary
            
        Returns:
            qa_dict: Dictionary containing question-answer pairs in the format {question: answer}
        """
        
        qa_dict = {}    
        
        # Iterate through the SQuAD data
        for article in raw_data["data"]:
            for paragraph in article["paragraphs"]:
                for qa in paragraph["qas"]:
                    question = qa["question"]
                    
                    if not qa["is_impossible"]:
                        if qa["answers"]:
                            answer = qa["answers"][0]["text"]
                            qa_dict[question] = answer
        
        return qa_dict

    @staticmethod
    def pdf_to_text(file_path: str) -> str:
        """
        Convert PDF file to text using the PdfConverter.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            text: Extracted text from the PDF file    
        """
        try:
            from marker.output import text_from_rendered
            converter = get_pdf_converter()
            rendered = converter(file_path)
            text, _, images = text_from_rendered(rendered)
        
            return text
        except Exception as e:
            print(f"Error converting PDF to text: {e}")
            return ""            


    def data_processing(self) -> str:
        """
        Process the raw data and extract question-answer pairs.
        
        Returns:
            str: Success message if processing is completed successfully.
        """
        try:
            files = [item for item in self.uploaded_files if item not in self.file_list]
            print(f"Files to process: {files}")
            logging.info(f"Files to process: {files}")
            
            for file in files:
                success = False
                print(f"Processing file: {file}")
                logging.info(f"Processing file: {file}")
                
                if file == "squad.json":
                    
                    # Load SQuAD dataset
                    raw_data = json.load(open("dags/data/squad.json", "r"))
                    qa_pairs = self.extract_qa_pairs(raw_data)
                    self.data_context[file] = qa_pairs
                    
                    if qa_pairs:
                        success = True
                        with open("dags/data/data_context.json", "w", encoding="utf-8") as f:
                            json.dump(self.data_context, f, ensure_ascii=False, indent=4)
                            print(f"Extracted {len(qa_pairs)} question-answer pairs from SQuAD dataset.")
                            logging.info(f"Extracted {len(qa_pairs)} question-answer pairs from SQuAD dataset.")
                
                elif file.endswith(".pdf"):
                    
                    file_path = f"dags/data/pdf/{file}"     
                    if os.path.isfile(file_path) == False:
                        print(f"File not found: {file_path}")
                        continue
                    
                    # Convert PDF to text                                
                    context = self.pdf_to_text(file_path)
                    print(f"Extracted text from {file}: {context[:100]}...")  # Print first 100 characters
                    
                    if context:
                        success = True
                        with open("dags/data/data_context.json", "w", encoding="utf-8") as f:
                            self.data_context[file] = {
                                "context": context
                            }
                            json.dump(self.data_context, f, ensure_ascii=False, indent=4)
                
                if success:      
                    # Update the config file
                    print(f"Successfully processed {file}")     
                    self.config_data["file_list"].append(file)
                    self.config_data["uploaded_files"].remove(file)
                    with open("dags/config.json", "w", encoding="utf-8") as f:
                        json.dump(self.config_data, f, ensure_ascii=False, indent=4)
            
        except Exception as e:
            print(f"Error processing data: {e}")
            return "Error processing data."
        finally:
            print("Data processing completed.")
            logging.info("Data processing completed.")
        
        return "Data processing completed successfully."


# if __name__ == "__main__":
#     processor = Data_Processing()
#     processor.data_processing()

# Download SQuAD dataset
# wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O squad.json