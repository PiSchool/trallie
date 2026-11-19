STARTER_SCHEMAS = [
  {
    "domain": "Government Report",
    "attributes": ["report_id","title","publication_date","issuing_agency","summary","keywords","recommendations","author_name","author_designation","author_department"]
  },
  {
    "domain": "Legal Document",
    "attributes": ["case_number","title","court_name","filing_date","verdict","citations","plaintiff","defendant","judge_name","judge_designation"]
  },
  {
    "domain": "Resume",
    "attributes": ["candidate_name","candidate_email","candidate_phone","candidate_address","experience_role","experience_company","experience_duration","experience_achievements","education_degree","education_institution","education_year","skill_name","skill_proficiency","certification_name","certification_issuer","certification_date"]
  },
  {
    "domain": "Invoice",
    "attributes": ["invoice_number","invoice_date","due_date","seller_name","buyer_name","tax","grand_total","payment_status","item_description","item_quantity","item_unit_price","item_total"]
  },
  {
    "domain": "Scientific Paper",
    "attributes": ["doi","title","abstract","keywords","journal","publication_date","author_name","author_affiliation","reference_citation_id","reference_title","reference_authors","reference_year","figure_id","figure_caption"]
  },
  {
    "domain": "Web Page",
    "attributes": ["url","title","meta_description","keywords","author","publish_date","body_text","link_href","link_anchor_text","media_src","media_alt","media_type"]
  },
  {
    "domain": "Incident Report",
    "attributes": ["incident_id","type","date","time","location","description","status","reporter_name","reporter_role","reporter_contact","action_description","action_date"]
  },
  {
    "domain": "Email",
    "attributes": ["message_id","subject","sender","date_sent","body_text","recipient_to","recipient_cc","recipient_bcc","attachment_filename","attachment_type","attachment_size"]
  },
  {
    "domain": "Medical Report",
    "attributes": ["patient_id","patient_name","patient_age","patient_gender","doctor_name","doctor_specialization","diagnosis_condition","diagnosis_severity","diagnosis_notes","medication_name","medication_dosage","medication_frequency","test_type","test_result","test_date","visit_date","visit_notes"]
  },
  {
    "domain": "Patent",
    "attributes": ["patent_number","title","abstract","filing_date","publication_date","assignee","inventor_name","inventor_affiliation","claim_number","claim_text","classification_code","classification_description"]
  },
  {
    "domain": "Product Catalog",
    "attributes": ["product_id","name","description","category","brand","price","discount","availability_status","stock_quantity"]
  },
  {
    "domain": "Financial Report",
    "attributes": ["report_id","filing_type","filing_date","company_name","cik","fiscal_year","period_end_date","auditor_name","balance_sheet_assets","balance_sheet_liabilities","balance_sheet_equity","income_statement_revenue","income_statement_expenses","income_statement_net_income","cash_flow_operating","cash_flow_investing","cash_flow_financing","earnings_per_share","notes_to_financials"]
  },
  {
    "domain": "Contract",
    "attributes": ["contract_id","title","effective_date","end_date","governing_law","party_1_name","party_1_role","party_2_name","party_2_role","payment_terms","obligations","termination_clause","confidentiality_clause","signature_party_1","signature_party_2","signature_date"]
  }
]
