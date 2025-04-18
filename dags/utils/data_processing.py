import os
import json
import logging
from typing import Dict


class Data_Processing:
    
    _pdf_converter = None
    
    def __init__(
        self, 
        config_path: str = "dags/config.json", 
        data_context_path: str = "dags/data/data_context.json"
    ):
        """
        Initialize the Data_Processing class.
        
        Args:
            config_path: Path to the config file
            data_context_path: Path to the data context file
        """
        self.config_path = config_path
        self.data_context_path = data_context_path
        try:
            self.config_data = json.load(open(self.config_path, "r"))
            self.data_context = json.load(open(self.data_context_path, "r"))
            self.uploaded_files = self.config_data.get("uploaded_files", [])
            self.file_list = self.config_data.get("file_list", [])
        except Exception as e:
            logging.error(f"Error loading config or data context: {e}")
            self.config_data = {}
            self.data_context = {}
            self.uploaded_files = []
            self.file_list = []

    @staticmethod
    def extract_squad_document(raw_data: Dict[str, str]) -> list:
        """
        Extract document from SQuAD formatted data.
        
        Args:
            raw_data: SQuAD formatted data dictionary
            
        Return:
            squad_document: List of documents extracted from the SQuAD formatted data
        """        
        squad_document = []   
        for item in raw_data['data']:
            for paragraph in item['paragraphs']:
                try:
                    squad_document.append(paragraph['context'])
                except KeyError:
                    logging.error("Warning: 'context' not found in paragraph.")
        
        return list(set(squad_document))
    
    @staticmethod
    def get_pdf_converter():
        """
        Get the PDF converter.
        
        Returns:
            pdf_converter: The PDF converter.
        """
        try:
            if Data_Processing._pdf_converter is None:
                from marker.converters.pdf import PdfConverter
                from marker.models import create_model_dict
                Data_Processing._pdf_converter = PdfConverter(
                    artifact_dict=create_model_dict(),
                )
            return Data_Processing._pdf_converter
        except Exception as e:
            logging.error(f"Error loading PDF converter: {e}")
            raise e

    def pdf_to_text(self, file_path: str) -> str:
        """
        Convert PDF file to text using the PdfConverter.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            text: Extracted text from the PDF file    
        """
        try:
            from marker.output import text_from_rendered
            converter = self.get_pdf_converter()
            rendered = converter(file_path)
            text, _, images = text_from_rendered(rendered)
        
            return text
        except Exception as e:
            logging.error(f"Error converting PDF to text: {e}")
            return ""
    
    @staticmethod    
    def markdown_text_splitter(context: str) -> list:
        """
        Split the text into smaller chunks based on Markdown headers.
        
        Args:
            context: Context to be split
            
        Returns:
            context_list: List of text chunks
        """
        try:
            from langchain_text_splitters import MarkdownHeaderTextSplitter
            
            context_list = []
            text_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3")
                ],
                strip_headers=False
            )
            splits = text_splitter.split_text(context)
            
            for text in splits:
                context_list.append(text.page_content)
            
            return context_list
        except Exception as e:
            logging.error(f"Error splitting text: {e}")
            return []
    
    def save_file(self, file: str, file_path: str, document: list = []) -> None:
        """
        Save the extracted document to a file.
        
        Args:
            file: Name of the file
            file_path: Path to the file
            document: List of documents to be saved
        """
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                if file == "squad.json":
                    self.data_context[file] = document
                    json.dump(self.data_context, f, ensure_ascii=False, indent=4)
                    logging.info(f"Extracted {len(document)} documents from SQuAD dataset.")
                elif file.endswith(".pdf"):
                    self.data_context[file] = document
                    json.dump(self.data_context, f, ensure_ascii=False, indent=4)
                    logging.info(f"Extracted {len(document)} documents from {file}.")
                elif file == "config":
                    json.dump(self.config_data, f, ensure_ascii=False, indent=4)
                    logging.info(f"Config file saved.")
        except Exception as e:
            logging.error(f"Error saving file {file}: {e}")
            raise e

    def data_processing(self) -> str:
        """
        Process the uploaded files and extract data from them.
        
        Returns:
            str: Success message if processing is completed successfully.
        """
        try:
            files = [item for item in self.uploaded_files if item not in self.file_list]
            logging.info(f"Files to process: {files}")
            
            for file in files:
                success = False
                logging.info(f"Processing file: {file}")
                
                if file == "squad.json":
                    
                    # Load SQuAD dataset
                    raw_data = json.load(open("dags/data/squad.json", "r"))
                    squad_document = self.extract_squad_document(raw_data)
                    
                    if squad_document:
                        success = True
                        self.save_file(file, self.data_context_path, squad_document[:10])
                
                elif file.endswith(".pdf"):
                    
                    file_path = f"dags/data/pdf/{file}"     
                    if os.path.isfile(file_path) == False:
                        logging.error(f"File not found: {file_path}")
                        continue
                    
                    # Convert PDF to text and split into chunks                          
                    context = self.pdf_to_text(file_path)
                    context_list = self.markdown_text_splitter(context)
                    
                    if context_list:
                        success = True
                        self.save_file(file, self.data_context_path, context_list)
                
                if success:      
                    # Update the config file
                    logging.info(f"Successfully processed {file}")     
                    self.config_data["file_list"].append(file)
                    self.config_data["uploaded_files"].remove(file)
            
            self.save_file("config", self.config_path)            
            return "Data processing completed successfully."            
        except Exception as e:
            logging.error(f"Error processing data: {e}")
            return "Error processing data."
        finally:
            logging.info("Data processing completed.")
            return "Data processing completed."
        


# Download SQuAD dataset
# wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O squad.json