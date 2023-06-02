#===========================================================
# ParseNewspaper.py
#===========================================================
# 
# This module parses newspapers and extracts sentences by 
# reference to companies 
#
# Inputs:   -sPathToCompaniesData, path to company keyword 
#            database (.csv file); str
#           -lHeaders, list of strings corresponding to 
#            headers for columns in the company keyword .csv
#            file, expecting in the following order:
#              -"Name"; header for Name column; most 
#               commonly used company name; single entry
#              -"Associated Names"; header for Associated
#                Names column; additional terms 
#                associated w/ company; multiple entry 
#                comma-seperated
#              -"Ticker"; header for Ticker column; 
#                company ticker; single entry
#              -"Stock Table Name"; header for Stock 
#                Table Name column; company name as used
#                in stock tables; single entry
#           -sPathToPDF, path to newspaper to parse (.pdf 
#            file); str
#           -sOutputFolder, path to folder for output file; 
#            str
#
# Outputs:  -Extracted news sentences and associated 
#            companies:
#             -as list of strings
#             -as .csv file
#
# To Run:   Setup:
#             >>from MikesLibrary.ParseNewspaper import ParseNews
#             >>sPathToCompaniesData = <path_to_company_naming_data_csv_file>
#             >>lHeaders = ["Name", "Associated Names", "Ticker", "Stock Table Name"]
#             >>sPathToPDF = <path_to_newspaper_pdf_file>
#             >>sOutputFolder = <path_to_output_folder>
#             
#           Instantiate ParseNews object:
#             >>myNews = ParseNews(sPathToCompaniesData, lHeaders)
#
#           Extract Company References from News:
#             >>lMyNews = myNews.extractText(sPathToPDF, sOutputFolder)
#
# TODO:     1. Collect extracted text as full articles, 
#              rather than carriage-return separated blocks 
#              of text
#
# Author:   Mike Thomas, May 2023
#
#===========================================================

#------------------------------------
# Python Imports
#------------------------------------
import re                                                           #Access Python Regular Expressions library
from datetime import datetime                                       #Access Python Datetime library
import warnings                                                     #Access Python library for user warnings
                                                                    
#--------------------------------
# 3rd Party Imports aka Dependencies
#--------------------------------
from MikesLibrary.HelperFunctions import stringNow                  #Access user-defined helper function to return a string with current datetime; for archiving
import pandas as pd                                                 #Access pandas library; for .csv file input/outpu
from pdfminer.high_level import extract_pages                       #Access extract_pages function from pdfminer.six library's common use-case functions in high_level.py; to extract text as page object from .pdf files
from pdfminer.layout import LTTextContainer                         #Access LTTextContainer class from pdfminer.six library's layout classes in layout.py; container for text objects

#---------------------------------------------------------
# ParseNews Class Definition
#---------------------------------------------------------
#
# Class to parse a newspaper for sentences referencing 
# companies
# 
# Methods:
#   -__init__ ->    initialize class instantiation
#   -extractText -> extract company referenced text from 
#                   news
#
#---------------------------------------------------------
class ParseNews():

    #--------------------------------------------------------
    # __init__ Method Definition
    #--------------------------------------------------------
    #
    # Class instatiation initialization method
    # 
    # Inputs:   -namesPath, path to the company names .csv 
    #            file; formated in columns with headers as 
    #            specified in columnHeaders, str
    #           -columnHeaders, list of strings corresponding
    #            to headers for columns in the company names
    #            .csv file, expecting in the following order:
    #              -"Name"; header for Name column; most 
    #               commonly used company name; single entry
    #              -"Associated Names"; header for Associated
    #                Names column; additional terms 
    #                associated w/ company; multiple entry 
    #                comma-seperated
    #              -"Ticker"; header for Ticker column; 
    #                company ticker; single entry
    #              -"Stock Table Name"; header for Stock 
    #                Table Name column; company name as used
    #                in stock tables; single entry
    #
    # Outputs:  -initialized instantiation of ParseText class
    #           
    #--------------------------------------------------------
    def __init__(self, namesPath, columnHeaders):
    
        #--------------------------------------------------------
        # Initialize object attributes
        #--------------------------------------------------------
        #
        # Save needed inputs to the instantiated ParseText object
        # as attributes 
        #
        #---------------------------------------------------------
        self.namesPath = namesPath
        self.nameHeader = columnHeaders[0]
        self.assocNamesHeader = columnHeaders[1]
        self.tickerHeader = columnHeaders[2]
        self.stockTableNameHeader = columnHeaders[3]
        
        #--------------------------------------------------------
        # Pull Company Data
        #--------------------------------------------------------
        #
        # Pull company data and filter out unwanted information; 
        # format as data frame
        #
        #---------------------------------------------------------
        dfCompanies = pd.read_csv(self.namesPath)                                                                                           #pull company data into pandas data frame
        dfCompanies = dfCompanies.filter(items=[self.nameHeader, self.assocNamesHeader, self.tickerHeader, self.stockTableNameHeader])      #retain only needed columns
        dfCompanies = dfCompanies.dropna(subset=[self.nameHeader])                                                                          #filter out rows where commonly used name is not available (aka drop NAs); unlikely to yield good results
        self.dfCompanies = dfCompanies                                                                                                      #set company data attribute for object instance; data frame

        #--------------------------------------------------------
        # Build Company Keywords
        #--------------------------------------------------------
        #
        # Build up a string of keywords associated with each 
        # company
        #
        #---------------------------------------------------------
        serAssocNames = self.dfCompanies[self.assocNamesHeader]                 #pull associated names column into a series
        serAssocNames = serAssocNames.dropna()                                  #drop cells w/ NaNs from the series
        serAssocNames = serAssocNames.str.replace(", ", "|", regex=False)       #perform simple string replace operation; swap comma-space for or, for regex pattern
        sAssocNames = serAssocNames.str.cat(sep="|")                            #concatenate all cells in the series into a string, use or as seperator, for regex pattern
        serNames = self.dfCompanies[self.nameHeader]                            #pull names column into a series
        sNames = serNames.str.cat(sep="|")                                      #concatenate all cells in the series into a string, use or as seperator, for regex pattern
        sKeywords = sNames + "|" + sAssocNames                                  #concatenate names and associated names into keywords string, use or as seperator, for regex pattern
        self.keywords = sKeywords                                               #set keywords attribute for object instance, str
        
        #--------------------------------------------------------
        # Build Company Keywords Search RegExp
        #--------------------------------------------------------
        #
        # Build up the regular expressions pattern from company 
        # keywords
        #
        # Regular Expression will follow the following pattern:
        #
        #                          Group 1                              Group 2                                                           Group 3
        # <start_of_string|whitespace|dash|em_length_dash><company_keyword|company_keyword><whitespace|comma|semi-colon|colon|period|dash|em_length_dash|apostrophe|right_single_quote_mark|end_of_string>
        #  
        # For Example: 
        # 
        # "(\A|\s|\-|—)(Apple|Alphabet|Google)(\s|,|;|:|\.|\-|—|'|’|\Z)"
        #
        #---------------------------------------------------------
        self.regexPattern = "(\A|\s|\-|—)(" + self.keywords + ")(\s|,|;|:|\.|\-|—|'|’|\Z)"
    
    #--------------------------------------------------------
    # extractText Method Definition
    #--------------------------------------------------------
    #
    # Method to extract company referenced text from the 
    # news; performs the following:
    #   -Extracts text elements from the .pdf file by page
    #   -Parses text elements for company keywords
    #   -Outputs text elements and company names to .csv file
    # 
    # Inputs:   -newsPath, path to the newspaper .pdf file,  
    #            str
    #           -outFolder, folder location to put output 
    #            file, str
    #
    # Outputs:  -lCompanyNews, string list with company names
    #            and referenced text from the .pdf file
    #           -.csv output file with company names and 
    #            referenced text from the .pdf file
    #
    # Note:     Extracting text takes ~3mins for a ~40-page
    #           .pdf
    #           
    #--------------------------------------------------------
    def extractText(self, newsPath, outFolder):
        
        #--------------------------------------------------------
        # Initialize Objects
        #--------------------------------------------------------
        startTime = datetime.now()                                  #pull current time; for code running time analysis
        lCompanyNews = []                                           #instantiate list to hold sentences referencing companies
        pageCount = 0                                               #instantiate page counter
        elCount = 0                                                 #instantiate element counter
        
        #---------------------------------------------------------
        # replaceDash Function Definition
        #---------------------------------------------------------
        #
        # Define Helper function for use with regular expression 
        # .sub function
        # 
        # Inputs:   RegEx Match; object of type objMatch
        #
        # Outputs:  Replacement string; based on match found
        #
        #---------------------------------------------------------
        def replaceDash(objMatch):
            if objMatch.group(0) == '-\n':                              #if matched object is <dash><carriage_return>
                return ''                                               #return <nothing>
            if objMatch.group(0) == '\n':                               #if matched object is <carriage_return>
                return ' '                                              #return <space>

        #--------------------------------------------------------
        # Extract Pages
        #--------------------------------------------------------
        #
        # Extract pages from .pdf using pdfminer.six 
        # extract_pages function; returns object of type page 
        # object generator
        #
        #---------------------------------------------------------
        print("Extracting news, this may take a few moments...")        #user info re extracting text
        pages = extract_pages(newsPath)                                 #extract text
        
        #--------------------------------------------------------
        # Loop Thru Extracted Pages
        #--------------------------------------------------------
        for page in pages:
            
            #--------------------------------------------------------
            # Increment Page Counter
            #--------------------------------------------------------
            pageCount+=1
            
            #--------------------------------------------------------
            # Loop Thru Elements on Page
            #--------------------------------------------------------
            for element in page:                                                                    
                
                #--------------------------------------------------------
                # If Text Container...
                #--------------------------------------------------------
                #
                # If this text element is a text container class...
                #
                #---------------------------------------------------------
                if isinstance(element, LTTextContainer):
                    
                    #--------------------------------------------------------
                    # Parse and Clean Text
                    #--------------------------------------------------------
                    tempText = element.get_text()                       #pull this text element's text
                    tempText = re.sub('-\n', replaceDash, tempText)     #use regex substitute to clean up dashes spliting words over carriage returns
                    tempText = re.sub('\n', replaceDash, tempText)      #use regex substitute to clean up carriage returns in general
                    tempText = re.sub("\ue013" + " ", "", tempText)     #use regex substitute to remove non-printable characters (replace diamond bullet and trailing space with nothing)
                    tempText = re.sub(chr(233), "e", tempText)          #use regex substitute to replace non-printable characters (replace small e and acute with e)
                    tempText = re.sub(chr(8212), "-", tempText)         #use regex substitute to replace non-printable characters (replace Em dash with dash)
                    tempText = re.sub(chr(8216), "'", tempText)         #use regex substitute to replace non-printable characters (replace left single quotation mark with single quote)
                    tempText = re.sub(chr(8217), "'", tempText)         #use regex substitute to replace non-printable characters (replace right single quotation mark with single quote)
                    tempText = re.sub(chr(8220), "'", tempText)         #use regex substitute to replace non-printable characters (replace left double quotation mark with single quote)
                    tempText = re.sub(chr(8221), "'", tempText)         #use regex substitute to replace non-printable characters (replace right double quotation mark with single quote)
                    tempText = re.sub(chr(8226), "-", tempText)         #use regex substitute to replace non-printable characters (replace bullet with dash)
                    tempText = re.sub(chr(8230), "...", tempText)       #use regex substitute to replace non-printable characters (replace horizontal ellipses with ...)
                    tempText = re.sub(chr(64257), "fi", tempText)       #use regex substitute to replace non-printable characters (replace small ligature Fi with fi)
                    
                    #--------------------------------------------------------
                    # Perform RegEX Search
                    #--------------------------------------------------------
                    #
                    # Parse this text element for company keywords using 
                    # regular expression findall() method with company names 
                    # patterns; returns a list of tuples (according to regex 
                    # pattern groups) of strings if found
                    #
                    #---------------------------------------------------------
                    lMatches = re.findall(self.regexPattern, tempText)
                    
                    #--------------------------------------------------------
                    # If Match Found 
                    #--------------------------------------------------------
                    if lMatches:
                    
                        #--------------------------------------------------------
                        # Instantiate Company Names List
                        #--------------------------------------------------------
                        #
                        # Instantiate a list of company names, to track which 
                        # companies have already been identified within this text
                        # block
                        #
                        #---------------------------------------------------------    
                        lCompanyNames = []

                        #--------------------------------------------------------
                        # Loop Thru Matches
                        #--------------------------------------------------------
                        #
                        # Loop thru the matches list
                        #
                        #---------------------------------------------------------    
                        for tThisMatch in lMatches:
                            
                            #--------------------------------------------------------
                            # Pull keyword
                            #--------------------------------------------------------
                            #
                            # Found result is returned as tuple of 3 objects based
                            # on groups within RegEx Pattern:
                            #
                            # <Group 1>, <Group 2>, <Group 3>
                            #
                            #---------------------------------------------------------
                            sLeftTemp, sThisKeyword, sRightTemp = tThisMatch
                        
                            #--------------------------------------------------------
                            # Check if this text is from the newspapper index
                            #--------------------------------------------------------
                            #
                            # If this match is from the newspaper's index; no useful
                            # content; don't process this text
                            #
                            #---------------------------------------------------------
                            if re.search("\.\.\.\.", tempText):                                         #if this text contains 4 dots in a row together...
                                #print(f"Found in index => {sThisKeyword}\n***")                        #for troubleshooting; user message found match in index
                                continue                                                                #proceed to next text element without processing
                            
                            #--------------------------------------------------------
                            # Determine Row Index
                            #--------------------------------------------------------
                            #
                            # Determine the row within the company naming data 
                            # associated with this found keyword
                            #
                            #---------------------------------------------------------
                            if True in (self.dfCompanies[self.assocNamesHeader].str.find(sThisKeyword) >= 0).tolist():                      #if this keyword is found in the associated names column...
                                 rowIndex = (self.dfCompanies[self.assocNamesHeader].str.find(sThisKeyword) >= 0).tolist().index(True)      #capture index for the row
                            elif True in (self.dfCompanies[self.nameHeader].str.find(sThisKeyword) >= 0).tolist():                          #else if this keyword is found in the name columns...
                                rowIndex = (self.dfCompanies[self.nameHeader].str.find(sThisKeyword) >= 0).tolist().index(True)             #capture index for the row
                            else:                                                                                                           #otherwise (this shouldnt happen...)
                                warnings.warn(f"Couldnt find {sThisKeyword} in companies data")                                             #user warning...couldnt find row index...
                        
                            #--------------------------------------------------------
                            # Check if this match is from the newspapper stock table
                            #--------------------------------------------------------
                            #
                            # If this match is from the newspaper's stock table; no 
                            # useful content; don't process this text
                            #
                            #---------------------------------------------------------
                            if rowIndex:                                                                                                                            #if row index found...
                                if type(self.dfCompanies[self.stockTableNameHeader][rowIndex]) == str:                                                              #if its stock table name cell is a str aka not NaN...
                                    sStockTable = self.dfCompanies[self.stockTableNameHeader][rowIndex] + " " + self.dfCompanies[self.tickerHeader][rowIndex]       #build a stock table search string: <company_stock_table_name> <company_ticker>
                                    if sStockTable in tempText:                                                                                                     #if stock table search string found in element text...
                                        #print(f"Found in stock table => {sThisKeyword}\n***")                                                                      #for troubleshooting; user message found match in stock table
                                        continue                                                                                                                    #proceed to next element without processing
                                
                            #--------------------------------------------------------
                            # Check if this match is a sentence fragment
                            #--------------------------------------------------------
                            #
                            # If this match is very short; no useful content; don't 
                            # process this text
                            #
                            #---------------------------------------------------------
                            if len(tempText) < len(sThisKeyword) + 6:                                   #if length of this element's text is only 6 chrs longer than the company keyword...
                                #print(f"Found sentence fragment => {sThisKeyword}\n***")               #for troubleshooting; user message found sentence fragment
                                continue                                                                #proceed to next element without processing
                            
                            #--------------------------------------------------------
                            # Process this text
                            #--------------------------------------------------------
                            #
                            # If none of the above catches fired, and if this company 
                            # hasnt already been identified with this text block; 
                            # process this text
                            #
                            #---------------------------------------------------------
                            companyName = self.dfCompanies[self.nameHeader][rowIndex]                   #pull company name
                            if companyName not in lCompanyNames:                                        #if this company is not already in the company list for this text block...
                                elCount+=1                                                              #increment element counter
                                #print(f"Found Keyword => {sThisKeyword}")                              #for troubleshooting; output company keyword found for user
                                #print(f"Keyword Associated with Company => {companyName}")             #for troubleshooting; output company name for user
                                #print(f"Page => {pageCount}")                                          #for troubleshooting; output page found on for user
                                #print(f"Element Text =>\n{tempText}\n***")                             #for troubleshooting; output element text for user
                                lCompanyNews.append([tempText, companyName])                            #capture this text block and this company name in data list
                                lCompanyNames.append(companyName)                                       #add this company name to the list of companies identified within this text block
        
        #--------------------------------------------------------
        # Output Company News
        #--------------------------------------------------------    
        if lCompanyNews:                                                                            #if company news found...
            sPathToOutput = outFolder + "\\" + stringNow() + "_Company_News.csv"                    #build path for output file
            dfCompanyNews = pd.DataFrame(lCompanyNews)                                              #convert list of company sentences to a pandas dataframe
            dfCompanyNews.to_csv(sPathToOutput, header=["news", "company"], index=False)            #output dataframe to output .csv; add column headers; don't output rows (aka index)
            print(f"Company News Saved to: {sPathToOutput}")                                        #user message; output save location
        else:                                                                                       #if no company news found (this may happen if .pdf not searchable...)
            warnings.warn("Couldnt find any company news")                                          #user warning...couldnt find news...
            warnings.warn(f"Length of lCompanyNews => {len(lCompanyNews)}")                         #user message; length of company news list
        
        #--------------------------------------------------------
        # Output Run Time Results
        #--------------------------------------------------------
        endTime = datetime.now()                                                                    #pull current time, for code running time analysis
        print(f"Run Results\nNumber of Elements with Company Keyword Found => {elCount}")           #user message; keywords found results
        print(f"Run Time => {(endTime-startTime).seconds} seconds")                                 #user message; code run time
        
        #--------------------------------------------------------
        # Return News
        #--------------------------------------------------------
        #
        # Method returns lCompanyNews; list of strings with news 
        # text and associated companies
        #
        #---------------------------------------------------------                
        return lCompanyNews