#===========================================================
# HelperFunctions.py
#===========================================================
# 
# This module is a collection of helper functions used
# in other modules
#
# Author:   Mike Thomas, May 2023
#
#===========================================================

#------------------------------------
# Python Imports
#------------------------------------
from datetime import datetime                                       #Access Python Datetime library
                                                                    
#---------------------------------------------------------
# stringNow Function Definition
#---------------------------------------------------------
#
# Helper function to return a string with datetime for now
# 
# Inputs:   None
#
# Outputs:  String formatted as YYYY_MM_DD_HHMM
#
#---------------------------------------------------------
def stringNow():

    #--------------------------------------------------------
    # Build date-time string 
    #--------------------------------------------------------
    #
    # Capture the current date and time; format as string
    #
    #---------------------------------------------------------
    dtNow = datetime.now()                                                  #pull now; datetime object
    sYear = str(dtNow.year)                                                 #convert year to string
    iMonth = dtNow.month                                                    #pull month; int
    if iMonth < 10:                                                         #if less than 10...
        sMonth = '0' + str(iMonth)                                          #convert to string and add a 0
    else:                                                                   #otherwise...
        sMonth = str(iMonth)                                                #just convert to string
    iDay = dtNow.day                                                        #pull day; int
    if iDay < 10:                                                           #if less than 10...
        sDay = '0' + str(iDay)                                              #convert to string and add a 0
    else:                                                                   #otherwise...
        sDay = str(iDay)                                                    #just convert to string
    iHour = dtNow.hour                                                      #pull hour; int
    if iHour < 10:                                                          #if less than 10...
        sHour = '0' + str(iHour)                                            #convert to string and add a 0
    else:                                                                   #otherwise...
        sHour = str(iHour)                                                  #just convert to string
    iMinute = dtNow.minute                                                  #pull minute; int
    if iMinute < 10:                                                        #if less than 10...
        sMinute = '0' + str(iMinute)                                        #convert to string and add a 0
    else:                                                                   #otherwise...
        sMinute = str(iMinute)                                              #just convert to string
    sNow = sYear + '_' + sMonth + '_' + sDay + '_' + sHour + sMinute        #Build string as YYYY_MM_DD_HHMM
    
    return sNow                                                             #return current date-time string