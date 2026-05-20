# PubMedQA Benchmark Eval

## Summary

- Dataset: `data/eval_pubmedqa_strict_90.json`
- Model: `qwen2.5:7b`
- Cases: 90
- Case pass rate: 0.578
- Label accuracy: 0.611
- Grounded status rate: 0.989
- Source hit@1: 0.978
- Source hit@3: 0.978
- Citation pass rate: 1.000
- Mean groundedness: 0.989
- Mean hallucination rate: 0.011
- Mean latency: 2763.2 ms

## Labels

| Label | Count | Accuracy |
|---|---:|---:|
| maybe | 30 | 0.600 |
| no | 30 | 0.633 |
| yes | 30 | 0.600 |

## Cases

| Case | Expected | Predicted | Hit@1 | Hit@3 | Citation | Pass | Top source |
|---|---|---|---:|---:|---:|---:|---|
| `pubmedqa-10223070` | maybe | maybe | True | True | True | True | 10223070 / Is perforation of the appendix a risk factor for tubal infertility and ectopic pregnancy? |
| `pubmedqa-10340286` | maybe | no | True | True | True | False | 10340286 / Is there a role for leukocyte and CRP measurements in the diagnosis of acute appendicitis in the elderly? |
| `pubmedqa-10605400` | maybe | maybe | True | True | True | True | 10605400 / Is the international normalised ratio (INR) reliable? |
| `pubmedqa-11034241` | maybe | maybe | True | True | True | True | 11034241 / Is routine intraoperative frozen-section examination of sentinel lymph nodes in breast cancer worthwhile? |
| `pubmedqa-11247896` | maybe | maybe | True | True | True | True | 11247896 / Can APC mutation analysis contribute to therapeutic decisions in familial adenomatous polyposis? |
| `pubmedqa-11296674` | maybe | maybe | True | True | True | True | 11296674 / Prostatic syndrome and pleural effusion: are they different diseases? |
| `pubmedqa-11411430` | maybe | yes | True | True | True | False | 11411430 / Antral follicle assessment as a tool for predicting outcome in IVF--is it a better predictor than age and FSH? |
| `pubmedqa-11458136` | maybe | no | True | True | True | False | 11458136 / Does managed care enable more low income persons to identify a usual source of care? |
| `pubmedqa-11570976` | maybe | maybe | False | False | True | False | 10223070 / Is perforation of the appendix a risk factor for tubal infertility and ectopic pregnancy? |
| `pubmedqa-11867487` | maybe | maybe | True | True | True | True | 11867487 / Does rugby headgear prevent concussion? |
| `pubmedqa-12407608` | maybe | yes | True | True | True | False | 12407608 / Does ultrasound imaging before puncture facilitate internal jugular vein cannulation? |
| `pubmedqa-12626177` | maybe | maybe | True | True | True | False | 12626177 / Can the Internet be used to improve sexual health awareness in web-wise young people? |
| `pubmedqa-12630042` | maybe | yes | True | True | True | False | 12630042 / Does body mass index (BMI) influence morbidity and long-term survival in gastric cancer patients after gastrectomy? |
| `pubmedqa-12790890` | maybe | maybe | True | True | True | True | 12790890 / Is the cell death in mesial temporal sclerosis apoptotic? |
| `pubmedqa-12805495` | maybe | maybe | True | True | True | True | 12805495 / Can patients be anticoagulated after intracerebral hemorrhage? |
| `pubmedqa-12920330` | maybe | maybe | True | True | True | True | 12920330 / Do somatic complaints predict subsequent symptoms of depression? |
| `pubmedqa-14599616` | maybe | maybe | True | True | True | True | 14599616 / Can a practicing surgeon detect early lymphedema reliably? |
| `pubmedqa-14992556` | maybe | maybe | True | True | True | True | 14992556 / Artefacts in 24-h pharyngeal and oesophageal pH monitoring: is simplification of pH data analysis feasible? |
| `pubmedqa-15381614` | maybe | no | True | True | True | False | 15381614 / Cutaneous melanoma in a multiethnic population: is this a different disease? |
| `pubmedqa-1571683` | maybe | maybe | True | True | True | True | 1571683 / Storage of vaccines in the community: weak link in the cold chain? |
| `pubmedqa-15787677` | maybe | no | True | True | True | False | 15787677 / Does aerobic fitness influence microvascular function in healthy adults at risk of developing Type 2 diabetes? |
| `pubmedqa-16361634` | maybe | maybe | True | True | True | True | 16361634 / Women with synchronous primary cancers of the endometrium and ovary: do they have Lynch syndrome? |
| `pubmedqa-16392897` | maybe | maybe | True | True | True | True | 16392897 / BCRABL transcript detection by quantitative real-time PCR : are correlated results possible from homebrew assays? |
| `pubmedqa-16538201` | maybe | yes | True | True | True | False | 16538201 / Does use of hydrophilic guidewires significantly improve technical success rates of peripheral PTA? |
| `pubmedqa-16735905` | maybe | yes | True | True | True | False | 16735905 / Does the severity of obstructive sleep apnea predict patients requiring high continuous positive airway pressure? |
| `pubmedqa-16816043` | maybe | yes | True | True | True | False | 16816043 / Do French lay people and health professionals find it acceptable to breach confidentiality to protect a patient's wife from a sexually transmitted disease? |
| `pubmedqa-16968876` | maybe | yes | True | True | True | False | 16968876 / Is a patient's self-reported health-related quality of life a prognostic factor for survival in non-small-cell lung cancer patients? |
| `pubmedqa-17076091` | maybe | maybe | True | True | True | False | 17076091 / Does obstructive sleep apnea affect aerobic fitness? |
| `pubmedqa-17076590` | maybe | yes | True | True | True | False | 17076590 / Counter sampling combined with medical provider education: do they alter prescribing behavior? |
| `pubmedqa-9140335` | maybe | maybe | True | True | True | True | 9140335 / Does fluoridation reduce the use of dental services among adults? |
| `pubmedqa-10375486` | no | no | True | True | True | True | 10375486 / Are variations in the use of carotid endarterectomy explained by population Need? |
| `pubmedqa-10381996` | no | no | True | True | True | True | 10381996 / Clinician assessment for acute chest syndrome in febrile patients with sickle cell disease: is it accurate enough? |
| `pubmedqa-10456814` | no | yes | True | True | True | False | 10456814 / Does desflurane alter left ventricular function when used to control surgical stimulation during aortic surgery? |
| `pubmedqa-10473855` | no | maybe | True | True | True | False | 10473855 / Is delayed gastric emptying following pancreaticoduodenectomy related to pylorus preservation? |
| `pubmedqa-10575390` | no | yes | True | True | True | False | 10575390 / Do follow-up recommendations for abnormal Papanicolaou smears influence patient adherence? |
| `pubmedqa-10593212` | no | no | True | True | True | True | 10593212 / Does loss of consciousness predict neuropsychological decrements after concussion? |
| `pubmedqa-10732884` | no | no | True | True | True | True | 10732884 / Does coronary angiography before emergency aortic surgery affect in-hospital mortality? |
| `pubmedqa-10757151` | no | no | True | True | True | True | 10757151 / Does ischemic preconditioning require reperfusion before index ischemia? |
| `pubmedqa-10781708` | no | maybe | True | True | True | False | 10781708 / Thrombosis prophylaxis in hospitalised medical patients: does prophylaxis in all patients make sense? |
| `pubmedqa-7482275` | no | maybe | True | True | True | False | 7482275 / Necrotizing fasciitis: an indication for hyperbaric oxygenation therapy? |
| `pubmedqa-7497757` | no | yes | True | True | True | False | 7497757 / Cardiopulmonary bypass temperature does not affect postoperative euthyroid sick syndrome? |
| `pubmedqa-7547656` | no | no | True | True | True | True | 7547656 / Does continuous intravenous infusion of low-concentration epinephrine impair uterine blood flow in pregnant ewes? |
| `pubmedqa-7664228` | no | maybe | True | True | True | False | 7664228 / Discharging patients earlier from Winnipeg hospitals: does it adversely affect quality of care? |
| `pubmedqa-8199520` | no | no | True | True | True | True | 8199520 / Are physicians meeting the needs of family caregivers of the frail elderly? |
| `pubmedqa-8200238` | no | no | True | True | True | True | 8200238 / Must early postoperative oral intake be limited to laparoscopy? |
| `pubmedqa-8422202` | no | no | True | True | True | True | 8422202 / Metered-dose inhalers. Do health care providers know what to teach? |
| `pubmedqa-8521557` | no | maybe | True | True | True | False | 8521557 / The insertion allele of the ACE gene I/D polymorphism. A candidate gene for insulin resistance? |
| `pubmedqa-8566975` | no | no | True | True | True | True | 8566975 / Serovar specific immunity to Neisseria gonorrhoeae: does it exist? |
| `pubmedqa-8738894` | no | yes | True | True | True | False | 8738894 / Diabetes mellitus among Swedish art glass workers--an effect of arsenic exposure? |
| `pubmedqa-8847047` | no | no | True | True | True | True | 8847047 / Prognosis of well differentiated small hepatocellular carcinoma--is well differentiated hepatocellular carcinoma clinically early cancer? |
| `pubmedqa-8921484` | no | yes | True | True | True | False | 8921484 / Does gestational age misclassification explain the difference in birthweights for Australian aborigines and whites? |
| `pubmedqa-9044116` | no | no | True | True | True | True | 9044116 / Biliary atresia: should all patients undergo a portoenterostomy? |
| `pubmedqa-9100537` | no | no | True | True | True | True | 9100537 / Can nonproliferative breast disease and proliferative breast disease without atypia be distinguished by fine-needle aspiration cytology? |
| `pubmedqa-9347843` | no | maybe | False | False | True | False | - / - |
| `pubmedqa-9444542` | no | no | True | True | True | True | 9444542 / Does hippocampal atrophy on MRI predict cognitive decline? |
| `pubmedqa-9446993` | no | no | True | True | True | True | 9446993 / Can dentists recognize manipulated digital radiographs? |
| `pubmedqa-9542484` | no | no | True | True | True | True | 9542484 / Does successful completion of the Perinatal Education Programme result in improved obstetric practice? |
| `pubmedqa-9602458` | no | no | True | True | True | True | 9602458 / Does the Child Health Computing System adequately identify children with cerebral palsy? |
| `pubmedqa-9603166` | no | no | True | True | True | True | 9603166 / Should all human immunodeficiency virus-infected patients with end-stage renal disease be excluded from transplantation? |
| `pubmedqa-9645785` | no | no | True | True | True | True | 9645785 / Is a mandatory general surgery rotation necessary in the surgical clerkship? |
| `pubmedqa-2224269` | yes | yes | True | True | True | True | 2224269 / Should general practitioners call patients by their first names? |
| `pubmedqa-2503176` | yes | yes | True | True | True | True | 2503176 / Inhibin: a new circulating marker of hydatidiform mole? |
| `pubmedqa-7860319` | yes | maybe | True | True | True | False | 7860319 / Measuring hospital mortality rates: are 30-day data enough? |
| `pubmedqa-8017535` | yes | yes | True | True | True | True | 8017535 / Substance use and HIV-related sexual behaviors among US high school students: are they related? |
| `pubmedqa-8111516` | yes | yes | True | True | True | True | 8111516 / Do family physicians make good sentinels for influenza? |
| `pubmedqa-8165771` | yes | yes | True | True | True | True | 8165771 / Ultrasound in squamous cell carcinoma of the penis; a useful addition to clinical staging? |
| `pubmedqa-8245806` | yes | maybe | True | True | True | False | 8245806 / Does family practice at residency teaching sites reflect community practice? |
| `pubmedqa-8262881` | yes | no | True | True | True | False | 8262881 / Body dysmorphic disorder: does it have a psychotic subtype? |
| `pubmedqa-8375607` | yes | yes | True | True | True | True | 8375607 / Is the breast best for children with a family history of atopy? |
| `pubmedqa-8910148` | yes | maybe | True | True | True | False | 8910148 / Transesophageal echocardiographic assessment of left ventricular function in brain-dead patients: are marginally acceptable hearts suitable for transplantation? |
| `pubmedqa-8916748` | yes | yes | True | True | True | True | 8916748 / Do socioeconomic differences in mortality persist after retirement? |
| `pubmedqa-8985020` | yes | yes | True | True | True | True | 8985020 / Does induction chemotherapy have a role in the management of nasopharyngeal carcinoma? |
| `pubmedqa-9003088` | yes | maybe | True | True | True | False | 9003088 / Immunohistochemical assessment of steroid hormone receptors in tissues of the anal canal. Implications for anal incontinence? |
| `pubmedqa-9107172` | yes | no | True | True | True | False | 9107172 / Bridge experience with long-term implantable left ventricular assist devices. Are they an alternative to transplantation? |
| `pubmedqa-9142039` | yes | no | True | True | True | False | 9142039 / Does pediatric housestaff experience influence tests ordered for infants in the neonatal intensive care unit? |
| `pubmedqa-9191526` | yes | yes | True | True | True | True | 9191526 / Multidisciplinary breast cancer clinics. Do they work? |
| `pubmedqa-9199905` | yes | yes | True | True | True | True | 9199905 / Vertical lines in distal esophageal mucosa (VLEM): a true endoscopic manifestation of esophagitis in children? |
| `pubmedqa-9278754` | yes | maybe | True | True | True | False | 9278754 / Are head and neck specific quality of life measures necessary? |
| `pubmedqa-9363244` | yes | no | True | True | True | False | 9363244 / Does occupational nuclear power plant radiation affect conception and pregnancy? |
| `pubmedqa-9363529` | yes | yes | True | True | True | True | 9363529 / Does psychological distress predict disability? |
| `pubmedqa-9381529` | yes | yes | True | True | True | True | 9381529 / Immune suppression by lysosomotropic amines and cyclosporine on T-cell responses to minor and major histocompatibility antigens: does synergy exist? |
| `pubmedqa-9427037` | yes | yes | True | True | True | True | 9427037 / Are endothelial cell patterns of astrocytomas indicative of grade? |
| `pubmedqa-9465206` | yes | maybe | True | True | True | False | 9465206 / "Occult" posttraumatic lesions of the knee: can magnetic resonance substitute for diagnostic arthroscopy? |
| `pubmedqa-9483814` | yes | no | True | True | True | False | 9483814 / Does para-cervical block offer additional advantages in abortion induction with gemeprost in the 2nd trimester? |
| `pubmedqa-9488747` | yes | yes | True | True | True | True | 9488747 / Syncope during bathing in infants, a pediatric form of water-induced urticaria? |
| `pubmedqa-9550200` | yes | yes | True | True | True | True | 9550200 / Does lunar position influence the time of delivery? |
| `pubmedqa-9569972` | yes | yes | True | True | True | True | 9569972 / Proliferative index obtained by DNA image cytometry. Does it add prognostic information in Auer IV breast cancer? |
| `pubmedqa-9582182` | yes | maybe | True | True | True | False | 9582182 / Does the SCL 90-R obsessive-compulsive dimension identify cognitive impairments? |
| `pubmedqa-9616411` | yes | yes | True | True | True | True | 9616411 / Do general practitioner hospitals reduce the utilisation of general hospital beds? |
| `pubmedqa-9722752` | yes | yes | True | True | True | True | 9722752 / Does bone anchor fixation improve the outcome of percutaneous bladder neck suspension in female stress urinary incontinence? |

## Config

- `candidate_k`: `20`
- `top_k`: `1`
- `temperature`: `0.0`
- `RAG_EVIDENCE_FILTER_ENABLED`: `true`
- `RAG_ANSWER_QUALITY_GATE_ENABLED`: `true`
- `RAG_CORPUS_VERSION`: `pubmedqa-strict-90-v1`
