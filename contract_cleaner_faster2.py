import re
import chardet
from solidity_parser import parser
import logging
import json
class SourceCodeCleanerAndFormatter:
    def __init__(self, input_file=None):
        self.input_file = input_file
        self.arr_soucrce = None
        self.source_code = None
        self.json_data = None
        self.located_remove = []
        self.save_remove = []
        self.infor_con = {}
        self.save_name = {}
        self.n_fun = 1
        self.n_var = 1
        self.n_con = 1

    def read_input_file(self):
        try:
            with open(self.input_file, 'rb') as file:
                raw_data = file.read()
                detected_encoding = chardet.detect(raw_data)['encoding']
                if detected_encoding:
                    self.source_code = raw_data.decode(detected_encoding)
                else:
                    # If encoding detection fails, fall back to UTF-8
                    self.source_code = raw_data.decode('utf-8')
        except Exception as e:
            # Handle the case where decoding still fails
            self.log_error('read_input_file', e)
            self.source_code = None  # Set the source_code to None to indicate an er
    def read_parse_file(self):
        ast_tree = parser.parse(self.source_code, loc=True)
        with open("parse.json", "w") as f:
            json.dump(ast_tree, f, indent= 4)
        with open("parse.json", "r") as json_file:
            self.json_data = json.load(json_file)["children"]
            self.arr_soucrce = self.source_code.split('\n')
    def replace_with_spaces(self, original_string, start_index, end_index):
        # Check if start_index and end_index are within the bounds of the string
        if start_index < 0 or end_index > len(original_string):
            print("Invalid indices")
            return original_string
        # Create a new string with the substring replaced by spaces
        new_string = original_string[:start_index] + ' ' * (end_index - start_index) + original_string[end_index:]

        return new_string
    def support_detect_remove(self, json_data, name):
        if isinstance(json_data, dict):
            for key, value in json_data.items():
                if key == 'name' and value in self.save_remove:
                    check = True
                    if value in self.infor_con[name]["fun"]:
                        check = False
                    for con in self.infor_con[name]["base_con"]:
                        if self.infor_con.get(con) and value in self.infor_con[con]["fun"]:
                            check = False
                    if check:
                        return True
                if isinstance(value, (dict, list)):
                    if self.support_detect_remove(value, name):
                        return True
        elif isinstance(json_data, list):
            for item in json_data:
                if isinstance(item, (dict, list)):
                    if self.support_detect_remove(item, name):
                        return True

        return False
    def detect_remove_and_get_name_convert(self,json_data, name):
        if isinstance(json_data, dict):
            if json_data.get("type") == "EventDefinition":
                self.save_remove.append(json_data["name"])
                self.located_remove.append((json_data["loc"]['start']['line'],json_data["loc"]['end']['line'],
                                           json_data["loc"]['start']['column'],json_data["loc"]['end']['column']))
                
            elif (json_data.get("stateMutability") == "pure" or json_data.get("stateMutability") == "view") or (json_data.get("type"
            ) == "FunctionDefinition" and len(json_data.get("body")) == 0) or (json_data.get("type") == "EmitStatement"):
                  self.located_remove.append((json_data["loc"]['start']['line'],json_data["loc"]['end']['line'],
                                           json_data["loc"]['start']['column'],json_data["loc"]['end']['column']))
        
            else:
                if (json_data.get("type") == "FunctionDefinition" or json_data.get("type") == "ModifierDefinition"
                    ) and json_data["name"] not in self.save_name and json_data["name"] != None:
                    self.infor_con[name]["fun"].append(json_data["name"])
                    self.save_name[json_data["name"]] = f"FUN{self.n_fun}"
                    self.n_fun += 1
                if (json_data.get("type") == "VariableDeclaration" or json_data.get("type") == "Parameter"
                    ) and json_data["name"] not in self.save_name and json_data["name"] != None:
                    self.save_name[json_data["name"]] = f"VAR{self.n_var}"
                    self.n_var += 1
                if json_data.get("type") == "ExpressionStatement" and self.support_detect_remove(json_data,name):
                    self.located_remove.append((json_data["loc"]['start']['line'],json_data["loc"]['end']['line'],
                                           json_data["loc"]['start']['column'],json_data["loc"]['end']['column']))

                for key, value in json_data.items():
                    # Check if the key represents a variable or function name
                    if isinstance(value, (dict, list)):
                        # Recursively process nested dictionaries and lists
                        self.detect_remove_and_get_name_convert(value, name)
        elif isinstance(json_data, list):
            for item in json_data:
                # Recursively process items in a list
                self.detect_remove_and_get_name_convert(item, name)
    def load_con(self, json_data):
        for con in json_data:
            if con:
                if con["name"] not in self.save_name:
                    self.save_name[con["name"]] = f"CON{self.n_con}"
                    self.n_con += 1
                self.infor_con[con["name"]] = {"fun":[], "base_con":[]}
                if con.get("baseContracts"):
                    for base in con["baseContracts"]:
                        self.infor_con[con["name"]]["base_con"].append(base["baseName"]["namePath"])
                self.detect_remove_and_get_name_convert(con, con["name"])       
    def replace_name(self):
        # Build a regular expression pattern to match the words outside of quotes
        pattern = '|'.join([rf'(?<!\")\b{re.escape(word)}\b(?!")' for word in self.save_name.keys()])

        # Define a function to handle the replacement based on the match
        def replace(match):
            return self.save_name[match.group(0)]
        # Use re.sub() with the defined pattern and replacement function
        self.source_code = re.sub(pattern, replace, self.source_code)
    def remove_pragma_import_library(self):
        # Remove pragma statements and import statements
        self.source_code = re.sub(r'^\s*pragma.*;', '', self.source_code, flags=re.MULTILINE)
        self.source_code = re.sub(r'^\s*import.*;', '', self.source_code, flags=re.MULTILINE)
        self.source_code = re.sub(r'^\s*import.*', '', self.source_code, flags=re.MULTILINE)
        #remove library
        library_pattern = r'library\s+\w+\s*{((?:(?:[^{}]+|{(?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*})*))}'
        self.source_code = re.sub(library_pattern, '', self.source_code, flags=re.DOTALL)
    def remove_comments(self):
        # remove all occurrences streamed comments (/*COMMENT */) from string
        self.source_code = re.sub(re.compile("/\*.*?\*/", re.DOTALL), "", self.source_code)
        # remove all occurrence single-line comments (//COMMENT\n ) from string
        self.source_code = re.sub(re.compile("//.*?\n"), "", self.source_code)

    def format_within_parentheses(self):
        pattern = r'\(([^()]*((?:\([^()]*\))[^()]*)*)\)'
        self.source_code = re.sub(pattern, lambda match: '(' + re.sub(
            r' {2,}', ' ', match.group(1).replace('\n', '')) + ')', self.source_code)
    def remove_redundant_line_breaks(self):
        lines = self.source_code.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        self.source_code = '\n'.join(cleaned_lines) 
    def remove_remainder(self):
        arr_source = self.source_code.split("\n")
        for loc in self.located_remove:
            if loc[0] == loc[1]:
                arr_source[loc[0]-1] = self.replace_with_spaces(arr_source[loc[0]-1],loc[2],loc[3]+1)
            else:
                arr_source[loc[0]-1] = self.replace_with_spaces(arr_source[loc[0]-1],loc[2],len(arr_source[loc[0]-1]))
                for i in range(loc[0],loc[1]-1):
                    arr_source[i] = ''
                arr_source[loc[1]-1] = self.replace_with_spaces(arr_source[loc[1]-1],0,loc[3]+1)
        self.source_code = '\n'.join(arr_source)                  
    def create_new_solFile(self):
        name_res = self.input_file.split('.')[0]
        with open(f"{name_res}_output.sol", "w") as f:
            f.write(self.source_code)
    def log_error(self, function_name, error):
        error_message = f"Error in {function_name} for file {self.input_file}: {error}\n"
        with open('error_log.txt', 'a', encoding='utf-8') as error_log_file:
            error_log_file.write(error_message)
        logging.error(error_message)
    def clean_source_code(self):
        try:
            self.remove_pragma_import_library()
            self.remove_comments()
            self.read_parse_file()
        except Exception as e:
            pass
            self.log_error('clean_source_code', e)

    def format_source_code(self):
        try:
            self.load_con(self.json_data)
            self.remove_remainder()
            self.replace_name()
            self.remove_redundant_line_breaks()
            self.format_within_parentheses()
        except Exception as e:
            self.log_error('format_source_code', e)