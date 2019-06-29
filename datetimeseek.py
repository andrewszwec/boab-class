import re

rxg_date_dmy = re.compile(r'(?i)((?P<day>[0123]?[0-9])?(?P<sep>\s|,|\-|\.)?(?P<month>((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w{0,6})|([01]?[0-9]))(?:\s|,|\-|\.)?(?P<year>([1,2]\d{3})|([1,2]\d)))')

rgx_date_mdy = re.compile(r'(?i)((?P<month>((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w{0,6})|([01]?[0-9]))(?P<sep>\s|,|\-|\.)?(?P<day>[0123]?[0-9])?(?:\s|,|\-|\.)?(?P<year>([1,2]\d{3})|([1,2]\d)))')

rgx_date_ymd = re.compile(r'(?i)((?P<year>([1,2]\d{3})|([1,2]\d)))(?P<sep>\s|,|\-|\.)?(?P<month>((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w{0,6})|([01]?[0-9]))(?:\s|,|\-|\.)?(?P<day>[0123]?[0-9])?')

rgx_date_ydm = re.compile(r'(?i)((?P<year>([1,2]\d{3})|([1,2]\d)))(?P<sep>\s|,|\-|\.)?(?P<day>[0123]?[0-9])?(?:\s|,|\-|\.)?(?P<month>((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w{0,6})|([01]?[0-9]))')


def __parsedate(matchobj,ordering):
    """Takes a date match object and an ordering returns a python datetime format string for the date"""
    dayformat = ""
    monthformat = ""
    yearformat = ""
    sep = ""
    year = matchobj.group("year")
    month = matchobj.group("month")
    day = matchobj.group("day")
    sep = "" if matchobj.group("sep") is None else matchobj.group("sep")

    dayformat = "%d" if day else ""

    if year:
        yearformat = "%y" if len(year)==4 else "%Y"
    else:
        yearformat = ""

    try:
        month = int(month)
        monthformat = "%m"
    except:
        monthformat = "%B" if len(month)>3 else "%b"

    if ordering == 'dmy':
        formatstring = dayformat+sep+monthformat+sep+yearformat
    elif ordering == 'mdy':
        formatstring = monthformat+sep+dayformat+sep+yearformat
    elif ordering == 'ymd':
        formatstring = yearformat+sep+monthformat+sep+dayformat
    elif ordering == 'ydm':
        formatstring = yearformat+sep+dayformat+sep+monthformat
    else:
        raise Exception("A date formating error occurred")

def seekdate(string):
    """Seeks for a something that looks like a date in a string and returns a format string if it finds one"""
    match = None
    if rxg_date_dmy.match(string):
        match = rxg_date_dmy.match(string)
        return __parsedate(match,'dmy')
    elif rgx_date_mdy.match(string):
        match = rgx_date_mdy.match(string)
        return  __parsedate(match,'myd')
    elif rgx_date_ymd.match(string):
        match = rgx_date_ymd.match(string)
        return __parsedate(match,'ymd')
    elif rgx_date_ydm.match(string):
        match = rgx_date_ydm.match(string)
        return __parsedate(match,'ydm')
    else:
        return None

def isdatecol(pd_series,sample_size=10):
    """Assesses whether the column contains dates and returns a datetime format string if it does. Otherwise returns a null"""
    pd_series_notnull = pd_series.dropna()
    samples = pd_series_notnull.astype(str).sample(sample_size,replace=True)

    dateformatstrings = samples.apply(seekdate)

    if dateformatstrings.isnull().any():
        return None
    else:
        return dateformatstrings.item()



