# InLegalNER gold schema
inlegalner_gold_entity_schema = {
    "lawyer": "lawyer mentioned in the document",
    "court": "court involved in the case",
    "judge": "judge presiding over the case",
    "petitioner": "person or entity filing the petition",
    "respondent": "person or entity responding to the petition",
    "case_number": "unique identifier for the case",
    "gpe": "geopolitical entity mentioned in the judgment",
    "date": "date of the judgment or case events",
    "org": "organization mentioned in the judgment",
    "statute": "legal statute referred to in the judgment",
    "witness": "person testifying in the case",
    "precedent": "previous case law referenced in the judgment",
    "provision": "specific provision of law cited",
    "other_person": "any other person mentioned in the judgment",
}

# MIT Restaurants NER gold data
restaurant_data_gold_schema = {
    "rating": "customer rating of the restaurant",
    "amenity": "facilities or services provided by the restaurant",
    "location": "geographical location of the restaurant",
    "restaurant_name": "name of the restaurant",
    "price": "cost range of the restaurant",
    "hours": "operating hours of the restaurant",
    "dish": "specific dish served by the restaurant",
    "cuisine": "type of cuisine offered by the restaurant",
}

# MIT Movies NER gold schema
movie_data_gold_schema = {
    "actor": "actor appearing in the movie",
    "plot": "summary or storyline of the movie",
    "opinion": "review or opinion about the movie",
    "award": "award won or nominated for the movie",
    "year": "release year of the movie",
    "genre": "category or genre of the movie",
    "origin": "country or region of the movie's production",
    "director": "director of the movie",
    "soundtrack": "music or soundtrack featured in the movie",
    "relationship": "character relationships in the movie",
    "character_name": "name of a character in the movie",
    "quote": "memorable quote from the movie",
}

# WNUT-17 NER gold schema
wnut17_entity_gold_schema = {
    "person": "an individual mentioned in the context",
    "location": "a geographical place or address",
    "group": "a collection of individuals or an organization",
    "corporation": "a legally recognized business entity",
    "product": "a manufactured or digital item",
    "creative-work": "a work of art, literature, film, or other creative content",
}

# CADEC Schema
medical_entity_schema = {
    "Adverse drug reaction": "adverse drug reaction caused by medication",
    "Disease": "a medical condition affecting health",
    "Drug": "a pharmaceutical substance used for treatment",
    "Finding": "a clinical or diagnostic observation",
    "Symptom": "a physical or mental indication of a condition",
}

# Industry data gold schema
industry_data_entity_gold_schema = {
    "location": "geographical location of the project",
    "industry": "sector or type of industry",
    "productionstatus": "current production status",
    "assettype": "type of asset",
    "productionvolume": "current production volume",
    "revenuegoal": "expected revenue",
    "investmentask": "amount of investment required",
    "propertyvalue": "value of the property or asset",
    "reserves": "amount of proven reserves",
}

# FDA Data Schema
medical_device_submission_schema = {
    "product code": "unique identifier for the medical device category",
    "purpose for submission": "reason for submitting the application",
    "predicate device name": "name of the previously approved device used for comparison",
    "type of test": "category of testing performed on the device",
    "measurand": "specific quantity or parameter measured by the device",
    "proprietary and established names": "brand and official names of the device",
    "proposed labeling": "suggested labeling and instructions for the device",
    "conclusion": "final assessment or decision regarding the device submission",
    "indications for use": "specific medical conditions the device is intended to address",
    "classification": "regulatory classification of the device",
    "applicant": "entity or individual submitting the application",
    "panel": "FDA advisory panel reviewing the submission",
    "intended use": "primary purpose and function of the device",
    "regulation section": "specific regulatory code governing the device",
    "510k number": "unique identifier for the 510(k) submission",
}
