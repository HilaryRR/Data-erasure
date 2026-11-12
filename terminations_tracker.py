import pandas as pd
import numpy as np
import re
import sys
from urllib.parse import urlparse

# Dataset imports
df = pd.read_csv("federal_terminations_110526.csv")
agency = pd.read_csv("agency_lookup.csv")

# Assigning an ID to each entry
df.insert(0, "id", [f"DI-{i:03d}" for i in range(len(df))])

# Column maintenance
df = df.rename(columns = {
    "Dataset Name":"dataset_name", "Acronym":"acronym", "Description":"description", "Caveats":"caveats", 
    "Agency":"agency", "Data Type":"topic", "Status":"status", "Date Terminated":"date_terminated", 
    "Link 1 URL":"link_1_url", "Link 1 Text":"link_1_text", "Link 1 Source":"link_1_source",
    "Link 2 URL":"link_2_url", "Link 2 Text":"link_2_text", "Link 2 Source":"link_2_source",
    "Link 3 URL":"link_3_url","Link 3 Text":"link_3_text", "Link 3 Source":"link_3_source",
    "Primary Link Wayback":"archive_url",
})
## Distinguishing between datasets and data elements
df["level"] = df["Section"].map({"Data Elements Removed":"data element", "Dataset Removed":"dataset"})
df = df.drop(columns=["Section"])

# Correcting datatypes
## Cleaning whitespace in text columns
def fix_text(s):
    if not isinstance(s, str):
        return s
    s = " ".join(s.split())
    try:
        s = s.encode("mac_roman").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return s or pd.NA
print(df.dtypes)
text_columns = df.select_dtypes(include = ["object", "string"]).columns
df[text_columns] = df[text_columns].apply(lambda col: col.map(fix_text))
## Standardizing date columns
df["date_terminated"] = pd.to_datetime(df["date_terminated"])

# URL handling
df = df.merge(agency, on = "agency", how = "left")
## Extracting domains from URLs 
link_columns = ["link_1_url", "link_2_url", "link_3_url", "archive_url"]
url_domains = df[link_columns].apply(lambda col: col.map(lambda url: urlparse(url).netloc if isinstance(url, str) else pd.NA))
## Identifying link type based on domain & making boolean columns
df["has_archive_link"] = url_domains.apply(lambda col: col.str.endswith(("archive.org", "archive.ph", "archive.is"))).any(axis = 1)
df["has_government_link"] = url_domains.apply(lambda col: col.str.endswith((".gov", ".mil"))).any(axis = 1)
 
# Source handling
df["evidence_class"] = np.select([df.has_government_link & df.has_archive_link, df.has_government_link, df.has_archive_link], ["Government and archive link", "Government link only", "Archive link only"], default = "Neither")

# Counting the occurrences of SOGI phrases
sogi_phrases = {
    "sex at birth": r"sex (?:assigned )?at birth",
    "transgender": r"transgender",
    "nonbinary": r"non[- ]?binary",
    "sexual orientation": r"sexual orientation",
    "gender identity": r"gender identity",
}
context_columns = ["description", "caveats"]
context = df[context_columns].fillna("").agg(" ".join, axis=1)
counts = pd.DataFrame({name: context.str.count(pattern, flags = re.IGNORECASE) for name, pattern in sogi_phrases.items()})
## Making boolean column for whether any SOGI phrase is mentioned 
df["mentions_sogi_phrase"] = counts.sum(axis = 1) > 0
## Making a separate dataframe for SOGI phrase counts in data elements
sogi = (counts[df.level == "data element"]
        .join(df[["id", "agency"]])
        .melt(id_vars=["id", "agency"],
              var_name="phrase",
              value_name="mention_count")
        .query("mention_count > 0")
        .sort_values(["phrase", "id"]))
 
columns = ["id", "dataset_name", "acronym", "description", "caveats",
           "agency", "agency_name", "parent_department", "topic", "level", "status", "date_terminated",
           "link_1_url", "link_1_text", "link_1_source", "link_2_url", "link_2_text", "link_2_source",
           "link_3_url", "link_3_text", "link_3_source", "archive_url",
           "has_government_link", "has_archive_link", "evidence_class", "mentions_sogi_phrase"]
df[columns].to_csv("federal_terminations_clean.csv", index = False)
sogi.to_csv("sogi_phrases.csv", index = False)

