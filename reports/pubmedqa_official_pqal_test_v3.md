# PubMedQA Benchmark Eval

## Summary

- Dataset: `/private/tmp/eval_pubmedqa_official_pqal_test.json`
- Model: `qwen2.5:7b`
- Cases: 500
- Case pass rate: 0.482
- Label accuracy: 0.496
- Grounded status rate: 0.992
- Source hit@1: 0.982
- Source hit@3: 0.982
- Citation pass rate: 0.998
- Mean groundedness: 0.982
- Mean hallucination rate: 0.018
- Mean latency: 2637.2 ms

## Labels

| Label | Count | Accuracy |
|---|---:|---:|
| maybe | 55 | 0.545 |
| no | 169 | 0.527 |
| yes | 276 | 0.467 |

## Cases

| Case | Expected | Predicted | Hit@1 | Hit@3 | Citation | Pass | Top source |
|---|---|---|---:|---:|---:|---:|---|
| `pubmedqa-official-12377809` | yes | maybe | True | True | True | False | 12377809 / Is anorectal endosonography valuable in dyschesia? |
| `pubmedqa-official-26163474` | yes | yes | True | True | True | True | 26163474 / Is there a connection between sublingual varices and hypertension? |
| `pubmedqa-official-19100463` | yes | yes | True | True | True | True | 19100463 / Is the affinity column-mediated immunoassay method suitable as an alternative to the microparticle enzyme immunoassay method as a blood tacrolimus assay? |
| `pubmedqa-official-18537964` | yes | maybe | True | True | True | False | 18537964 / Does a physician's specialty influence the recording of medication history in patients' case notes? |
| `pubmedqa-official-12913878` | yes | yes | True | True | True | True | 12913878 / Locoregional opening of the rodent blood-brain barrier for paclitaxel using Nd:YAG laser-induced thermo therapy: a new concept of adjuvant glioma therapy? |
| `pubmedqa-official-12765819` | yes | yes | True | True | True | True | 12765819 / Spinal subdural hematoma: a sequela of a ruptured intracranial aneurysm? |
| `pubmedqa-official-25475395` | yes | maybe | True | True | True | False | 25475395 / Is there a correlation between androgens and sexual desire in women? |
| `pubmedqa-official-19130332` | yes | yes | True | True | True | True | 19130332 / Is the zeolite hemostatic agent beneficial in reducing blood loss during arterial injury? |
| `pubmedqa-official-9427037` | yes | yes | True | True | True | True | 9427037 / Are endothelial cell patterns of astrocytomas indicative of grade? |
| `pubmedqa-official-24481006` | yes | yes | True | True | True | True | 24481006 / Should cavitation in proximal surfaces be reported in cone beam computed tomography examination? |
| `pubmedqa-official-8165771` | yes | yes | True | True | True | True | 8165771 / Ultrasound in squamous cell carcinoma of the penis; a useful addition to clinical staging? |
| `pubmedqa-official-22680064` | yes | yes | True | True | True | True | 22680064 / Can third trimester ultrasound predict the presentation of the first twin at delivery? |
| `pubmedqa-official-22540518` | yes | yes | True | True | True | True | 22540518 / Is micro-computed tomography reliable to determine the microstructure of the maxillary alveolar bone? |
| `pubmedqa-official-20629769` | yes | yes | True | True | True | True | 20629769 / Is primary angioplasty an acceptable alternative to thrombolysis? |
| `pubmedqa-official-21726930` | yes | maybe | True | True | True | False | 21726930 / Is endometrial polyp formation associated with increased expression of vascular endothelial growth factor and transforming growth factor-beta1? |
| `pubmedqa-official-21481154` | yes | yes | True | True | True | True | 21481154 / Improvements in survival of gynaecological cancer in the Anglia region of England: are these an effect of centralisation of care and use of multidisciplinary management? |
| `pubmedqa-official-22902073` | yes | yes | True | True | True | True | 22902073 / Estimated fetal weight by ultrasound: a modifiable risk factor for cesarean delivery? |
| `pubmedqa-official-26370095` | yes | maybe | True | True | True | False | 26370095 / Are financial incentives cost-effective to support smoking cessation during pregnancy? |
| `pubmedqa-official-18041059` | yes | yes | True | True | True | True | 18041059 / Do adjuvant aromatase inhibitors increase the cardiovascular risk in postmenopausal women with early breast cancer? |
| `pubmedqa-official-15041506` | yes | yes | True | True | True | True | 15041506 / Is fear of anaphylactic shock discouraging surgeons from more widely adopting percutaneous and laparoscopic techniques in the treatment of liver hydatid cyst? |
| `pubmedqa-official-11146778` | yes | maybe | True | True | True | False | 11146778 / Risk stratification in emergency surgical patients: is the APACHE II score a reliable marker of physiological impairment? |
| `pubmedqa-official-27281318` | yes | no | True | True | True | False | 27281318 / Can Flexible Instruments Create Adequate Femoral Tunnel Lengths at 90° of Knee Flexion in Anterior Cruciate Ligament Reconstruction? |
| `pubmedqa-official-21645374` | yes | maybe | True | True | True | False | 21645374 / Do mitochondria play a role in remodelling lace plant leaves during programmed cell death? |
| `pubmedqa-official-9465206` | yes | maybe | True | True | True | False | 9465206 / "Occult" posttraumatic lesions of the knee: can magnetic resonance substitute for diagnostic arthroscopy? |
| `pubmedqa-official-25887165` | yes | yes | True | True | True | True | 25887165 / Does Sensation Return to the Nasal Tip After Microfat Grafting? |
| `pubmedqa-official-15995461` | yes | yes | True | True | False | False | 15995461 / Do some U.S. states have higher/lower injury mortality rates than others? |
| `pubmedqa-official-21850494` | yes | maybe | True | True | True | False | 21850494 / Hepatorenal syndrome: are we missing some prognostic factors? |
| `pubmedqa-official-19106867` | yes | yes | True | True | True | True | 19106867 / The Main Gate Syndrome: a new format in mass-casualty victim "surge" management? |
| `pubmedqa-official-21342862` | yes | maybe | True | True | True | False | 21342862 / Is EQ-5D a valid quality of life instrument in patients with acute coronary syndrome? |
| `pubmedqa-official-24352924` | yes | maybe | True | True | True | False | 24352924 / Is portable ultrasonography accurate in the evaluation of Schanz pin placement during extremity fracture fixation in austere environments? |
| `pubmedqa-official-16147837` | yes | maybe | True | True | True | False | 16147837 / Is grandmultiparity an independent risk factor for adverse perinatal outcomes? |
| `pubmedqa-official-26879871` | yes | maybe | True | True | True | False | 26879871 / Does depression diagnosis and antidepressant prescribing vary by location? |
| `pubmedqa-official-15918864` | yes | maybe | True | True | True | False | 15918864 / Learning needs of postpartum women: does socioeconomic status matter? |
| `pubmedqa-official-22075911` | yes | yes | True | True | True | True | 22075911 / Is there a differential in the dental health of new recruits to the British Armed Forces? |
| `pubmedqa-official-11035130` | yes | maybe | True | True | True | False | 11035130 / Do patients with rheumatoid arthritis established on methotrexate and folic acid 5 mg daily need to continue folic acid supplements long term? |
| `pubmedqa-official-21228436` | yes | maybe | False | False | True | False | 27549226 / Impact of MPH programs: contributing to health system strengthening in low- and middle-income countries? |
| `pubmedqa-official-11833948` | yes | maybe | True | True | True | False | 11833948 / Does a delay in transfer to a rehabilitation unit for older people affect outcome after fracture of the proximal femur? |
| `pubmedqa-official-17682349` | yes | no | True | True | True | False | 17682349 / Are there gender differences in the reasons why African Americans delay in seeking medical help for symptoms of an acute myocardial infarction? |
| `pubmedqa-official-17355582` | yes | yes | True | True | True | True | 17355582 / Does ambulatory process of care predict health-related quality of life outcomes for patients with chronic disease? |
| `pubmedqa-official-15597845` | yes | maybe | True | True | True | False | 15597845 / Is the combination with 2-methoxyestradiol able to reduce the dosages of chemotherapeutices in the treatment of human ovarian cancer? |
| `pubmedqa-official-10158597` | yes | yes | True | True | True | True | 10158597 / Does a dedicated discharge coordinator improve the quality of hospital discharge? |
| `pubmedqa-official-27549226` | yes | maybe | True | True | True | False | 27549226 / Impact of MPH programs: contributing to health system strengthening in low- and middle-income countries? |
| `pubmedqa-official-26348845` | yes | maybe | True | True | True | False | 26348845 / Pap smears with glandular cell abnormalities: Are they detected by rapid prescreening? |
| `pubmedqa-official-25588461` | yes | maybe | True | True | True | False | 25588461 / Can transcranial direct current stimulation be useful in differentiating unresponsive wakefulness syndrome from minimally conscious state patients? |
| `pubmedqa-official-23359100` | yes | yes | True | True | True | True | 23359100 / Is etoricoxib effective in preventing heterotopic ossification after primary total hip arthroplasty? |
| `pubmedqa-official-26548832` | yes | no | True | True | True | False | 26548832 / Assessing Patient Reported Outcomes Measures via Phone Interviews Versus Patient Self-Survey in the Clinic: Are We Measuring the Same Thing? |
| `pubmedqa-official-25756710` | yes | yes | True | True | True | True | 25756710 / Can emergency physicians accurately and reliably assess acute vertigo in the emergency department? |
| `pubmedqa-official-20297950` | yes | maybe | True | True | True | False | 20297950 / Proof of concept study: does fenofibrate have a role in sleep apnoea syndrome? |
| `pubmedqa-official-24622801` | yes | yes | True | True | True | True | 24622801 / Does implant coating with antibacterial-loaded hydrogel reduce bacterial colonization and biofilm formation in vitro? |
| `pubmedqa-official-9722752` | yes | yes | True | True | True | True | 9722752 / Does bone anchor fixation improve the outcome of percutaneous bladder neck suspension in female stress urinary incontinence? |
| `pubmedqa-official-20577124` | yes | yes | True | True | True | True | 20577124 / Is leptin involved in phagocytic NADPH oxidase overactivity in obesity? |
| `pubmedqa-official-19027440` | yes | yes | True | True | True | True | 19027440 / Can we predict which head and neck cancer survivors develop fears of recurrence? |
| `pubmedqa-official-18239988` | yes | yes | True | True | True | True | 18239988 / Differentiation of nonalcoholic from alcoholic steatohepatitis: are routine laboratory markers useful? |
| `pubmedqa-official-27858166` | yes | yes | True | True | True | True | 27858166 / Traumatic aortic injury: does the anatomy of the aortic arch influence aortic trauma severity? |
| `pubmedqa-official-27050489` | yes | maybe | True | True | True | False | 27050489 / The Prevalence of Incidentally Detected Idiopathic Misty Mesentery on Multidetector Computed Tomography: Can Obesity Be the Triggering Cause? |
| `pubmedqa-official-16266387` | yes | no | True | True | True | False | 16266387 / Fast foods - are they a risk factor for asthma? |
| `pubmedqa-official-27287237` | yes | yes | True | True | True | True | 27287237 / Assessment of appropriate antimicrobial prescribing: do experts agree? |
| `pubmedqa-official-11079675` | yes | yes | True | True | True | True | 11079675 / Pulmonary valve replacement in adults late after repair of tetralogy of fallot: are we operating too late? |
| `pubmedqa-official-10331115` | yes | no | True | True | True | False | 10331115 / Is amoxapine an atypical antipsychotic? |
| `pubmedqa-official-18594195` | yes | maybe | True | True | True | False | 18594195 / Do older patients who refuse to participate in a self-management intervention in the Netherlands differ from older patients who agree to participate? |
| `pubmedqa-official-22497340` | yes | maybe | True | True | True | False | 22497340 / Is horizontal semicircular canal ocular reflex influenced by otolith organs input? |
| `pubmedqa-official-16769333` | yes | maybe | True | True | True | False | 16769333 / Preoperative tracheobronchoscopy in newborns with esophageal atresia: does it matter? |
| `pubmedqa-official-20571467` | yes | no | True | True | True | False | 20571467 / Is it appropriate to implant kidneys from elderly donors in young recipients? |
| `pubmedqa-official-12094116` | yes | maybe | True | True | True | False | 12094116 / Is muscle power related to running speed with changes of direction? |
| `pubmedqa-official-17276182` | yes | no | True | True | True | False | 17276182 / Stretch-sensitive KCNQ1 mutation A link between genetic and environmental factors in the pathogenesis of atrial fibrillation? |
| `pubmedqa-official-26419377` | yes | yes | True | True | True | True | 26419377 / Are pelvic anatomical structures in danger during arthroscopic acetabular labral repair? |
| `pubmedqa-official-23810330` | yes | yes | True | True | True | True | 23810330 / Is intraoperative neuromonitoring associated with better functional outcome in patients undergoing open TME? |
| `pubmedqa-official-15151701` | yes | maybe | True | True | True | False | 15151701 / Profiling quality of care: Is there a role for peer review? |
| `pubmedqa-official-23736032` | yes | maybe | True | True | True | False | 23736032 / Multidisciplinary decisions in breast cancer: does the patient receive what the team has recommended? |
| `pubmedqa-official-28143468` | yes | maybe | True | True | True | False | 28143468 / Are performance measurement systems useful? |
| `pubmedqa-official-23495128` | yes | maybe | True | True | True | False | 23495128 / The colour of pain: can patients use colour to describe osteoarthritis pain? |
| `pubmedqa-official-12121321` | yes | yes | True | True | True | True | 12121321 / Do mossy fibers release GABA? |
| `pubmedqa-official-18570208` | yes | yes | True | True | True | True | 18570208 / Is severe macrosomia manifested at 11-14 weeks of gestation? |
| `pubmedqa-official-28707539` | yes | yes | True | True | True | True | 28707539 / Visceral adipose tissue area measurement at a single level: can it represent visceral adipose tissue volume? |
| `pubmedqa-official-22117569` | yes | yes | True | True | True | True | 22117569 / Is an advance care planning model feasible in community palliative care? |
| `pubmedqa-official-18783922` | yes | no | True | True | True | False | 18783922 / Do cytokines have any role in epilepsy? |
| `pubmedqa-official-15528969` | yes | yes | True | True | True | True | 15528969 / Is expert breast pathology assessment necessary for the management of ductal carcinoma in situ ? |
| `pubmedqa-official-19482903` | yes | no | True | True | True | False | 19482903 / Treadmill testing of children who have spina bifida and are ambulatory: does peak oxygen uptake reflect maximum oxygen uptake? |
| `pubmedqa-official-11977907` | yes | yes | True | True | True | True | 11977907 / Subclavian steal syndrome: can the blood pressure difference between arms predict the severity of steal? |
| `pubmedqa-official-17306983` | yes | yes | True | True | True | True | 17306983 / Is size-reducing ascending aortoplasty with external reinforcement an option in modern aortic surgery? |
| `pubmedqa-official-24318956` | yes | yes | True | True | True | True | 24318956 / Is digoxin use for cardiovascular disease associated with risk of prostate cancer? |
| `pubmedqa-official-22266735` | yes | maybe | True | True | True | False | 22266735 / Screening for gestational diabetes mellitus: are the criteria proposed by the international association of the Diabetes and Pregnancy Study Groups cost-effective? |
| `pubmedqa-official-22453060` | yes | no | True | True | True | False | 22453060 / Does a 4 diagram manual enable laypersons to operate the Laryngeal Mask Supreme®? |
| `pubmedqa-official-10401824` | yes | maybe | True | True | True | False | 10401824 / Is laparoscopic reoperation for failed antireflux surgery feasible? |
| `pubmedqa-official-15208005` | yes | yes | True | True | True | True | 15208005 / The Omega-3 Index: a new risk factor for death from coronary heart disease? |
| `pubmedqa-official-16713745` | yes | no | True | True | True | False | 16713745 / Do cytokine concentrations in pancreatic juice predict the presence of pancreatic diseases? |
| `pubmedqa-official-21864397` | yes | maybe | True | True | True | False | 21864397 / Factors determining the survival of nasopharyngeal carcinoma with lung metastasis alone: does combined modality treatment benefit? |
| `pubmedqa-official-25810292` | yes | yes | True | True | True | True | 25810292 / Is minimally invasive mitral valve repair with artificial chords reproducible and applicable in routine surgery? |
| `pubmedqa-official-11943048` | yes | maybe | True | True | True | False | 11943048 / Does receipt of hospice care in nursing homes improve the management of pain at the end of life? |
| `pubmedqa-official-23347337` | yes | no | True | True | True | False | 23347337 / Is intensive chemotherapy safe for rural cancer patients? |
| `pubmedqa-official-23992109` | yes | yes | True | True | True | True | 23992109 / Is the urinary biomarkers assessment a non-invasive approach to tubular lesions of the solitary kidney? |
| `pubmedqa-official-10922093` | yes | no | True | True | True | False | 10922093 / Does open access endoscopy close the door to an adequately informed patient? |
| `pubmedqa-official-26601554` | yes | yes | True | True | True | True | 26601554 / Do viral infections have a role in benign paroxysmal positional vertigo? |
| `pubmedqa-official-15489384` | yes | maybe | True | True | True | False | 15489384 / Does reducing spasticity translate into functional benefit? |
| `pubmedqa-official-27818079` | yes | yes | True | True | True | True | 27818079 / Is the Retromandibular Transparotid Approach a Reliable Option for the Surgical Treatment of Condylar Fractures? |
| `pubmedqa-official-24340838` | yes | yes | True | True | True | True | 24340838 / Do ventricular arrhythmias in athletes subside over time? |
| `pubmedqa-official-16971978` | yes | yes | True | True | True | True | 16971978 / Are complex coronary lesions more frequent in patients with diabetes mellitus? |
| `pubmedqa-official-21689015` | yes | maybe | True | True | True | False | 21689015 / Can dogs prime autistic children for therapy? |
| `pubmedqa-official-12846929` | yes | no | True | True | True | False | 12846929 / Quality of life in lung cancer patients: does socioeconomic status matter? |
| `pubmedqa-official-22694248` | yes | yes | True | True | True | True | 22694248 / Is there a model to teach and practice retroperitoneoscopic nephrectomy? |
| `pubmedqa-official-15488260` | yes | maybe | True | True | True | False | 15488260 / Is the type of remission after a major depressive episode an important risk factor to relapses in a 4-year follow up? |
| `pubmedqa-official-23690198` | yes | maybe | False | False | True | False | 26370095 / Are financial incentives cost-effective to support smoking cessation during pregnancy? |
| `pubmedqa-official-10173769` | yes | maybe | True | True | True | False | 10173769 / Longer term quality of life and outcome in stroke patients: is the Barthel index alone an adequate measure of outcome? |
| `pubmedqa-official-12098035` | yes | yes | True | True | True | True | 12098035 / Does a special interest in laparoscopy affect the treatment of acute cholecystitis? |
| `pubmedqa-official-23448747` | yes | yes | True | True | True | True | 23448747 / Do older adults with cancer fall more often? |
| `pubmedqa-official-24359102` | yes | maybe | True | True | True | False | 24359102 / Two-year follow-up survey of patients with allergic contact dermatitis from an occupational cohort: is the prognosis dependent on the omnipresence of the allergen? |
| `pubmedqa-official-14697414` | yes | maybe | True | True | True | False | 14697414 / Is there a favorable subset of patients with prostate cancer who develop oligometastases? |
| `pubmedqa-official-15050326` | yes | yes | True | True | True | True | 15050326 / Does radiotherapy around the time of pregnancy for Hodgkin's disease modify the risk of breast cancer? |
| `pubmedqa-official-27131771` | yes | no | True | True | True | False | 27131771 / Does left atrial appendage (LAA) occlusion device alter the echocardiography and electrocardiogram parameters in patients with atrial fibrillation? |
| `pubmedqa-official-26923375` | yes | yes | True | True | True | True | 26923375 / Is non-invasive diagnosis of esophageal varices in patients with compensated hepatic cirrhosis possible by duplex Doppler ultrasonography? |
| `pubmedqa-official-15841770` | yes | yes | True | True | True | True | 15841770 / Do inhaled steroids differ from cromones in terms of hospital admission rates for asthma in children? |
| `pubmedqa-official-25503376` | yes | maybe | True | True | True | False | 25503376 / Does airway surgery lower serum lipid levels in obstructive sleep apnea patients? |
| `pubmedqa-official-19394934` | yes | maybe | True | True | True | False | 19394934 / Israeli hospital preparedness for terrorism-related multiple casualty incidents: can the surge capacity and injury severity distribution be better predicted? |
| `pubmedqa-official-22188074` | yes | yes | True | True | True | True | 22188074 / Do instrumental activities of daily living predict dementia at 1- and 2-year follow-up? |
| `pubmedqa-official-21394762` | yes | yes | True | True | True | True | 21394762 / Is pelvic pain associated with defecatory symptoms in women with pelvic organ prolapse? |
| `pubmedqa-official-9582182` | yes | maybe | True | True | True | False | 9582182 / Does the SCL 90-R obsessive-compulsive dimension identify cognitive impairments? |
| `pubmedqa-official-28056802` | yes | maybe | True | True | True | False | 28056802 / Is non-HDL-cholesterol a better predictor of long-term outcome in patients after acute myocardial infarction compared to LDL-cholesterol? |
| `pubmedqa-official-18182265` | yes | yes | True | True | True | True | 18182265 / Body diffusion-weighted MR imaging of uterine endometrial cancer: is it helpful in the detection of cancer in nonenhanced MR imaging? |
| `pubmedqa-official-9142039` | yes | no | True | True | True | False | 9142039 / Does pediatric housestaff experience influence tests ordered for infants in the neonatal intensive care unit? |
| `pubmedqa-official-20084845` | yes | maybe | True | True | True | False | 20084845 / Biomolecular identification of allergenic pollen: a new perspective for aerobiological monitoring? |
| `pubmedqa-official-24298614` | yes | yes | True | True | True | True | 24298614 / Is the 7th TNM edition suitable for biological predictor in early gastric cancer? |
| `pubmedqa-official-12145243` | yes | maybe | True | True | True | False | 12145243 / Are lower fasting plasma glucose levels at diagnosis of type 2 diabetes associated with improved outcomes? |
| `pubmedqa-official-21952349` | yes | maybe | True | True | True | False | 21952349 / Remote ischemic postconditioning: does it protect against ischemic damage in percutaneous coronary revascularization? |
| `pubmedqa-official-27592038` | yes | maybe | True | True | True | False | 27592038 / Does multi-modal cervical physical therapy improve tinnitus in patients with cervicogenic somatic tinnitus? |
| `pubmedqa-official-25481573` | yes | yes | True | True | True | True | 25481573 / Processing fluency effects: can the content and presentation of participant information sheets influence recruitment and participation for an antenatal intervention? |
| `pubmedqa-official-20306735` | yes | yes | True | True | True | False | 20306735 / Fulfilling human resources development goal in West Africa: can the training of ophthalmologist diplomates be improved? |
| `pubmedqa-official-26864326` | yes | no | True | True | True | False | 26864326 / Predicting admission at triage: are nurses better than a simple objective score? |
| `pubmedqa-official-21123461` | yes | maybe | True | True | True | False | 21123461 / Are adult body circumferences associated with height? |
| `pubmedqa-official-12963175` | yes | yes | True | True | True | True | 12963175 / Can progression of valvar aortic stenosis be predicted accurately? |
| `pubmedqa-official-10548670` | yes | yes | True | True | True | True | 10548670 / Does the National Institutes of Health Stroke Scale favor left hemisphere strokes? |
| `pubmedqa-official-21848798` | yes | yes | True | True | True | True | 21848798 / MiraLAX vs. Golytely: is there a significant difference in the adenoma detection rate? |
| `pubmedqa-official-25675614` | yes | yes | True | True | True | True | 25675614 / Can gingival crevicular blood be relied upon for assessment of blood glucose level? |
| `pubmedqa-official-25986020` | yes | yes | True | True | True | True | 25986020 / Is zero central line-associated bloodstream infection rate sustainable? |
| `pubmedqa-official-18472368` | yes | yes | True | True | True | True | 18472368 / Does treatment duration affect outcome after radiotherapy for prostate cancer? |
| `pubmedqa-official-26578404` | yes | yes | True | True | True | True | 26578404 / Patient-Controlled Therapy of Breathlessness in Palliative Care: A New Therapeutic Concept for Opioid Administration? |
| `pubmedqa-official-14872327` | yes | yes | True | True | True | True | 14872327 / Is pain a clinically relevant problem in general adult psychiatry? |
| `pubmedqa-official-23412195` | yes | yes | True | True | True | True | 23412195 / Should displaced midshaft clavicular fractures be treated surgically? |
| `pubmedqa-official-24139705` | yes | maybe | True | True | True | False | 24139705 / Telemedicine and type 1 diabetes: is technology per se sufficient to improve glycaemic control? |
| `pubmedqa-official-23224030` | yes | maybe | True | True | True | False | 23224030 / Do European people with type 1 diabetes consume a high atherogenic diet? |
| `pubmedqa-official-24013712` | yes | maybe | True | True | True | False | 24013712 / Preoperative platelet count in esophageal squamous cell carcinoma: is it a prognostic factor? |
| `pubmedqa-official-15943725` | yes | maybe | True | True | True | False | 15943725 / Should serum pancreatic lipase replace serum amylase as a biomarker of acute pancreatitis? |
| `pubmedqa-official-27456836` | yes | yes | True | True | True | True | 27456836 / Do Electrochemiluminescence Assays Improve Prediction of Time to Type 1 Diabetes in Autoantibody-Positive TrialNet Subjects? |
| `pubmedqa-official-24671913` | yes | yes | True | True | True | True | 24671913 / Does SYNTAX score predict in-hospital outcomes in patients with ST elevation myocardial infarction undergoing primary percutaneous coronary intervention? |
| `pubmedqa-official-22825590` | yes | yes | True | True | True | True | 22825590 / Are behavioural risk factors to be blamed for the conversion from optimal blood pressure to hypertensive status in Black South Africans? |
| `pubmedqa-official-23361217` | yes | yes | True | True | True | True | 23361217 / Does the type of tibial component affect mechanical alignment in unicompartmental knee replacement? |
| `pubmedqa-official-18307476` | yes | maybe | True | True | True | False | 18307476 / Upstream solutions: does the supplemental security income program reduce disability in the elderly? |
| `pubmedqa-official-22237146` | yes | maybe | True | True | True | False | 22237146 / Can serum be used for analyzing the EGFR mutation status in patients with advanced non-small cell lung cancer? |
| `pubmedqa-official-25043083` | yes | maybe | True | True | True | False | 25043083 / Are failures of anterior cruciate ligament reconstruction associated with steep posterior tibial slopes? |
| `pubmedqa-official-26561905` | yes | no | True | True | True | False | 26561905 / Do teleoncology models of care enable safe delivery of chemotherapy in rural towns? |
| `pubmedqa-official-23517744` | yes | no | True | True | True | False | 23517744 / Is solitary kidney really more resistant to ischemia? |
| `pubmedqa-official-27136599` | yes | maybe | True | True | True | False | 27136599 / Is it safe to perform rectal anastomosis in gynaecological debulking surgery without a diverting stoma? |
| `pubmedqa-official-10749257` | yes | maybe | False | False | True | False | 28143468 / Are performance measurement systems useful? |
| `pubmedqa-official-17598882` | yes | maybe | True | True | True | False | 17598882 / Is breast cancer prognosis inherited? |
| `pubmedqa-official-15223779` | yes | maybe | True | True | True | False | 15223779 / c-Kit-dependent growth of uveal melanoma cells: a potential therapeutic target? |
| `pubmedqa-official-16776337` | yes | yes | True | True | True | True | 16776337 / Pituitary apoplexy: do histological features influence the clinical presentation and outcome? |
| `pubmedqa-official-23916653` | yes | yes | True | True | True | True | 23916653 / Orthostatic myoclonus: an underrecognized cause of unsteadiness? |
| `pubmedqa-official-10201555` | yes | yes | True | True | True | True | 10201555 / Is low serum chloride level a risk factor for cardiovascular mortality? |
| `pubmedqa-official-24751724` | yes | maybe | True | True | True | False | 24751724 / Does ethnicity affect where people with cancer die? |
| `pubmedqa-official-8910148` | yes | maybe | True | True | True | False | 8910148 / Transesophageal echocardiographic assessment of left ventricular function in brain-dead patients: are marginally acceptable hearts suitable for transplantation? |
| `pubmedqa-official-18065862` | yes | yes | True | True | True | True | 18065862 / Can the postoperative pain level be predicted preoperatively? |
| `pubmedqa-official-22617083` | yes | no | True | True | True | False | 22617083 / Does age moderate the effect of personality disorder on coping style in psychiatric inpatients? |
| `pubmedqa-official-25499207` | yes | maybe | True | True | True | False | 25499207 / Is neck pain associated with worse health-related quality of life 6 months later? |
| `pubmedqa-official-16465002` | yes | maybe | True | True | True | False | 16465002 / Dose end-tidal carbon dioxide measurement correlate with arterial carbon dioxide in extremely low birth weight infants in the first week of life? |
| `pubmedqa-official-25940336` | yes | yes | True | True | True | True | 25940336 / Does Residency Selection Criteria Predict Performance in Orthopaedic Surgery Residency? |
| `pubmedqa-official-24191126` | yes | yes | True | True | True | True | 24191126 / Is CA72-4 a useful biomarker in differential diagnosis between ovarian endometrioma and epithelial ovarian cancer? |
| `pubmedqa-official-8375607` | yes | yes | True | True | True | True | 8375607 / Is the breast best for children with a family history of atopy? |
| `pubmedqa-official-26965932` | yes | yes | True | True | True | True | 26965932 / Is Bare-Metal Stent Implantation Still Justifiable in High Bleeding Risk Patients Undergoing Percutaneous Coronary Intervention? |
| `pubmedqa-official-22012962` | yes | maybe | True | True | True | False | 22012962 / Marital status, living arrangement and mortality: does the association vary by gender? |
| `pubmedqa-official-12442934` | yes | yes | True | True | True | True | 12442934 / Does ibuprofen increase perioperative blood loss during hip arthroplasty? |
| `pubmedqa-official-19430778` | yes | maybe | True | True | True | False | 19430778 / Can magnetic resonance imaging accurately predict concordant pain provocation during provocative disc injection? |
| `pubmedqa-official-20605051` | yes | no | True | True | True | False | 20605051 / Does case-mix based reimbursement stimulate the development of process-oriented care delivery? |
| `pubmedqa-official-19108857` | yes | maybe | True | True | True | False | 19108857 / Cerebromediastinal tuberculosis in a child with a probable Say-Barber-Miller syndrome: a causative link? |
| `pubmedqa-official-24516646` | yes | yes | True | True | True | True | 24516646 / Is the determination of specific IgE against components using ISAC 112 a reproducible technique? |
| `pubmedqa-official-25752725` | yes | maybe | True | True | True | False | 25752725 / Schizophrenia patients with high intelligence: A clinically distinct sub-type of schizophrenia? |
| `pubmedqa-official-20537205` | yes | maybe | False | False | True | False | 10223070 / Is perforation of the appendix a risk factor for tubal infertility and ectopic pregnancy? |
| `pubmedqa-official-20602784` | yes | yes | True | True | True | True | 20602784 / Identification of racial disparities in breast cancer mortality: does scale matter? |
| `pubmedqa-official-22302761` | yes | yes | True | True | True | True | 22302761 / Can fractional lasers enhance transdermal absorption of topical lidocaine in an in vivo animal model? |
| `pubmedqa-official-18322741` | yes | yes | True | True | True | True | 18322741 / Does laparoscopic surgery decrease the risk of atrial fibrillation after foregut surgery? |
| `pubmedqa-official-14692023` | yes | maybe | True | True | True | False | 14692023 / Is breast cancer survival improving? |
| `pubmedqa-official-22348433` | yes | maybe | True | True | True | False | 22348433 / Does partial expander deflation exacerbate the adverse effects of radiotherapy in two-stage breast reconstruction? |
| `pubmedqa-official-26215326` | yes | yes | True | True | True | True | 26215326 / Does the clinical presentation of a prior preterm birth predict risk in a subsequent pregnancy? |
| `pubmedqa-official-23539689` | yes | maybe | True | True | True | False | 23539689 / Cold preparation use in young children after FDA warnings: do concerns still exist? |
| `pubmedqa-official-9363244` | yes | no | True | True | True | False | 9363244 / Does occupational nuclear power plant radiation affect conception and pregnancy? |
| `pubmedqa-official-24507422` | yes | maybe | True | True | True | False | 24507422 / Can shape analysis differentiate free-floating internal carotid artery thrombus from atherosclerotic plaque in patients evaluated with CTA for stroke or transient ischemic attack? |
| `pubmedqa-official-22350859` | yes | no | True | True | True | False | 22350859 / Can pictorial warning labels on cigarette packages address smoking-related health disparities? |
| `pubmedqa-official-19640728` | yes | maybe | True | True | True | False | 19640728 / Surgical treatment of prosthetic valve endocarditis in patients with double prostheses: is single-valve replacement safe? |
| `pubmedqa-official-23806388` | yes | maybe | True | True | True | False | 23806388 / Do nomograms designed to predict biochemical recurrence (BCR) do a better job of predicting more clinically relevant prostate cancer outcomes than BCR? |
| `pubmedqa-official-9920954` | yes | yes | True | True | True | True | 9920954 / Do "America's Best Hospitals" perform better for acute myocardial infarction? |
| `pubmedqa-official-8916748` | yes | yes | True | True | True | True | 8916748 / Do socioeconomic differences in mortality persist after retirement? |
| `pubmedqa-official-11970923` | yes | maybe | True | True | True | False | 11970923 / Convulsions and retinal haemorrhage: should we look further? |
| `pubmedqa-official-19302863` | yes | no | True | True | True | False | 19302863 / Is the use of cyanoacrylate in intestinal anastomosis a good and reliable alternative? |
| `pubmedqa-official-22532370` | yes | yes | True | True | True | True | 22532370 / Are increased carotid artery pulsatility and resistance indexes early signs of vascular abnormalities in young obese males? |
| `pubmedqa-official-18179827` | yes | no | True | True | True | False | 18179827 / Does topical ropivacaine reduce the post-tonsillectomy morbidity in pediatric patients? |
| `pubmedqa-official-18399830` | yes | maybe | True | True | True | False | 18399830 / Is robotically assisted laparoscopic radical prostatectomy less invasive than retropubic radical prostatectomy? |
| `pubmedqa-official-12595848` | yes | maybe | True | True | True | False | 12595848 / Is specialty care associated with improved survival of patients with congestive heart failure? |
| `pubmedqa-official-18158048` | yes | maybe | True | True | True | False | 18158048 / Histologic evaluation of the testicular remnant associated with the vanishing testes syndrome: is surgical management necessary? |
| `pubmedqa-official-23848044` | yes | yes | True | True | True | True | 23848044 / Does oxybutynin hydrochloride cause arrhythmia in children with bladder dysfunction? |
| `pubmedqa-official-11481172` | yes | maybe | True | True | True | False | 11481172 / Does the manic/mixed episode distinction in bipolar disorder patients run true over time? |
| `pubmedqa-official-28247485` | yes | yes | True | True | True | True | 28247485 / Is the first urinary albumin/creatinine ratio (ACR) in women with suspected preeclampsia a prognostic factor for maternal and neonatal adverse outcome? |
| `pubmedqa-official-24977765` | yes | maybe | True | True | True | False | 24977765 / Are pediatric concussion patients compliant with discharge instructions? |
| `pubmedqa-official-14551704` | yes | yes | True | True | True | True | 14551704 / Can communication with terminally ill patients be taught? |
| `pubmedqa-official-12632437` | yes | yes | True | True | True | True | 12632437 / Are environmental factors important in primary systemic vasculitis? |
| `pubmedqa-official-20850631` | yes | no | True | True | True | False | 20850631 / Diagnosis and follow-up in constipated children: should we use ultrasound? |
| `pubmedqa-official-17565137` | yes | no | True | True | True | False | 17565137 / Out of the smokescreen II: will an advertisement targeting the tobacco industry affect young people's perception of smoking in movies and their intention to smoke? |
| `pubmedqa-official-9616411` | yes | yes | True | True | True | True | 9616411 / Do general practitioner hospitals reduce the utilisation of general hospital beds? |
| `pubmedqa-official-22720085` | yes | yes | True | True | True | True | 22720085 / Does insulin resistance drive the association between hyperglycemia and cardiovascular risk? |
| `pubmedqa-official-21074975` | yes | maybe | True | True | True | False | 21074975 / Ultra high risk (UHR) for psychosis criteria: are there different levels of risk for transition to psychosis? |
| `pubmedqa-official-25604390` | yes | maybe | True | True | True | False | 25604390 / Aberrant loss of dickkopf-3 in gastric cancer: can it predict lymph node metastasis preoperatively? |
| `pubmedqa-official-14968373` | yes | maybe | True | True | True | False | 14968373 / Can CT predict the level of CSF block in tuberculous hydrocephalus? |
| `pubmedqa-official-10135926` | yes | no | True | True | True | False | 10135926 / Is oral endotracheal intubation efficacy impaired in the helicopter environment? |
| `pubmedqa-official-19419587` | yes | yes | True | True | True | True | 19419587 / Sternal plating for primary and secondary sternal closure; can it improve sternal stability? |
| `pubmedqa-official-23379759` | yes | yes | True | True | True | True | 23379759 / Can early second-look tympanoplasty reduce the rate of conversion to modified radical mastoidectomy? |
| `pubmedqa-official-19923859` | yes | no | True | True | True | False | 19923859 / Can T-cell deficiency affect spatial learning ability following toluene exposure? |
| `pubmedqa-official-22656647` | yes | maybe | True | True | True | False | 22656647 / Are acceptance rates of a national preventive home visit programme for older people socially imbalanced? |
| `pubmedqa-official-12163782` | yes | yes | True | True | True | True | 12163782 / Increased neutrophil migratory activity after major trauma: a factor in the etiology of acute respiratory distress syndrome? |
| `pubmedqa-official-21658267` | yes | maybe | True | True | True | False | 21658267 / Do improvements in outreach, clinical, and family and community-based services predict improvements in child survival? |
| `pubmedqa-official-9199905` | yes | yes | True | True | True | True | 9199905 / Vertical lines in distal esophageal mucosa (VLEM): a true endoscopic manifestation of esophagitis in children? |
| `pubmedqa-official-23375036` | yes | yes | True | True | True | True | 23375036 / An HIV1/2 point of care test on sputum for screening TB/HIV co-infection in Central India - Will it work? |
| `pubmedqa-official-24495711` | yes | maybe | True | True | True | False | 24495711 / Is crime associated with over-the-counter pharmacy syringe sales? |
| `pubmedqa-official-26820719` | yes | maybe | True | True | True | False | 26820719 / Colorectal cancer in young patients: is it a distinct clinical entity? |
| `pubmedqa-official-26516021` | yes | maybe | True | True | True | False | 26516021 / Does evidence-based practice improve patient outcomes? |
| `pubmedqa-official-20064872` | yes | yes | True | True | True | True | 20064872 / Can the prognosis of polymyalgia rheumatica be predicted at disease onset? |
| `pubmedqa-official-15708048` | yes | yes | True | True | True | True | 15708048 / Does prior benign prostate biopsy predict outcome for patients treated with radical perineal prostatectomy? |
| `pubmedqa-official-29112560` | yes | no | True | True | True | False | 29112560 / Is the Distance Worth It? |
| `pubmedqa-official-23949294` | yes | maybe | True | True | True | False | 23949294 / Treatment as prevention in resource-limited settings: is it feasible to maintain HIV viral load suppression over time? |
| `pubmedqa-official-10877371` | yes | no | True | True | True | False | 10877371 / Does head positioning influence anterior chamber depth in pseudoexfoliation syndrome? |
| `pubmedqa-official-23870157` | yes | yes | True | True | True | True | 23870157 / Are intraoperative precursor events associated with postoperative major adverse events? |
| `pubmedqa-official-18540901` | yes | yes | True | True | True | True | 18540901 / Transient tachypnea of the newborn (TTN): a role for polymorphisms in the beta-adrenergic receptor (ADRB) encoding genes? |
| `pubmedqa-official-21420186` | yes | maybe | True | True | True | False | 21420186 / Could ADMA levels in young adults born preterm predict an early endothelial dysfunction? |
| `pubmedqa-official-12484580` | yes | yes | True | True | True | True | 12484580 / Informed consent for total hip arthroplasty: does a written information sheet improve recall by patients? |
| `pubmedqa-official-23321509` | yes | yes | True | True | True | True | 23321509 / Quaternary cytoreductive surgery in ovarian cancer: does surgical effort still matter? |
| `pubmedqa-official-26907557` | yes | yes | True | True | True | True | 26907557 / Can a Novel Surgical Approach to the Temporomandibular Joint Improve Access and Reduce Complications? |
| `pubmedqa-official-22644412` | yes | yes | True | True | True | True | 22644412 / Hepatic arterial embolization for unresectable hepatocellular carcinomas: do technical factors affect prognosis? |
| `pubmedqa-official-25521278` | yes | yes | True | True | True | True | 25521278 / Is plate clearing a risk factor for obesity? |
| `pubmedqa-official-21845457` | yes | yes | True | True | True | True | 21845457 / Outcomes of severely injured adult trauma patients in an Australian health service: does trauma center level make a difference? |
| `pubmedqa-official-18565233` | yes | maybe | True | True | True | False | 18565233 / Does the lipid-lowering peroxisome proliferator-activated receptors ligand bezafibrate prevent colon cancer in patients with coronary artery disease? |
| `pubmedqa-official-17894828` | yes | no | True | True | True | False | 17894828 / Serum angiotensin-converting enzyme and frequency of severe hypoglycaemia in Type 1 diabetes: does a relationship exist? |
| `pubmedqa-official-10490564` | yes | maybe | True | True | True | False | 10490564 / Hypotension in patients with coronary disease: can profound hypotensive events cause myocardial ischaemic events? |
| `pubmedqa-official-7860319` | yes | maybe | True | True | True | False | 7860319 / Measuring hospital mortality rates: are 30-day data enough? |
| `pubmedqa-official-18568239` | yes | yes | True | True | True | True | 18568239 / Is the ability to perform transurethral resection of the prostate influenced by the surgeon's previous experience? |
| `pubmedqa-official-9488747` | yes | yes | True | True | True | True | 9488747 / Syncope during bathing in infants, a pediatric form of water-induced urticaria? |
| `pubmedqa-official-20354380` | yes | yes | True | True | True | True | 20354380 / Do women residents delay childbearing due to perceived career threats? |
| `pubmedqa-official-24245816` | yes | yes | True | True | True | True | 24245816 / Is trabecular bone related to primary stability of miniscrews? |
| `pubmedqa-official-11481599` | yes | yes | True | True | True | True | 11481599 / Acute respiratory distress syndrome in children with malignancy--can we predict outcome? |
| `pubmedqa-official-27217036` | yes | maybe | True | True | True | False | 27217036 / Neoadjuvant Imatinib in Locally Advanced Gastrointestinal stromal Tumours, Will Kit Mutation Analysis Be a Pathfinder? |
| `pubmedqa-official-23283159` | yes | maybe | True | True | True | False | 23283159 / Is obesity a risk factor for wheezing among adolescents? |
| `pubmedqa-official-19593710` | yes | maybe | True | True | True | False | 19593710 / Could ESC (Electronic Stability Control) change the way we drive? |
| `pubmedqa-official-18693227` | yes | maybe | True | True | True | False | 18693227 / Does a geriatric oncology consultation modify the cancer treatment plan for elderly patients? |
| `pubmedqa-official-21346501` | yes | yes | True | True | True | True | 21346501 / Can students' scores on preclerkship clinical performance examinations predict that they will fail a senior clinical performance examination? |
| `pubmedqa-official-17910536` | yes | yes | True | True | True | True | 17910536 / Adults with mild intellectual disabilities: can their reading comprehension ability be improved? |
| `pubmedqa-official-26304701` | yes | no | True | True | True | False | 26304701 / Can nurse-led preoperative education reduce anxiety and postoperative complications of patients undergoing cardiac surgery? |
| `pubmedqa-official-18616781` | yes | maybe | True | True | True | False | 18616781 / Is there a relationship between homocysteine and vitiligo? |
| `pubmedqa-official-9483814` | yes | no | True | True | True | False | 9483814 / Does para-cervical block offer additional advantages in abortion induction with gemeprost in the 2nd trimester? |
| `pubmedqa-official-12848629` | yes | maybe | True | True | True | False | 12848629 / Is a 9-month treatment sufficient in tuberculous enterocolitis? |
| `pubmedqa-official-25280365` | yes | maybe | True | True | True | False | 25280365 / Reporting and interpreting red blood cell morphology: is there discordance between clinical pathologists and clinicians? |
| `pubmedqa-official-25311479` | yes | yes | True | True | True | True | 25311479 / The inverse equity hypothesis: does it apply to coverage of cancer screening in middle-income countries? |
| `pubmedqa-official-16046584` | yes | maybe | True | True | True | False | 16046584 / Menopausal hormone therapy and irregular endometrial bleeding: a potential role for uterine natural killer cells? |
| `pubmedqa-official-26418441` | yes | yes | True | True | True | True | 26418441 / Can we ease the financial burden of colonoscopy? |
| `pubmedqa-official-22683044` | yes | maybe | True | True | True | False | 22683044 / Does open access publishing increase the impact of scientific articles? |
| `pubmedqa-official-26200172` | yes | yes | True | True | True | True | 26200172 / Can biofeedback training of psychophysiological responses enhance athletes' sport performance? |
| `pubmedqa-official-20121683` | yes | yes | True | True | True | True | 20121683 / Are patients willing participants in the new wave of community-based medical education in regional and rural Australia? |
| `pubmedqa-official-18222909` | yes | maybe | True | True | True | False | 18222909 / Are pectins involved in cold acclimation and de-acclimation of winter oil-seed rape plants? |
| `pubmedqa-official-12221908` | yes | yes | True | True | True | True | 12221908 / The HELPP syndrome--evidence of a possible systemic inflammatory response in pre-eclampsia? |
| `pubmedqa-official-24014276` | yes | maybe | True | True | True | False | 24014276 / Optimism and survival: does an optimistic outlook predict better survival at advanced ages? |
| `pubmedqa-official-24270957` | yes | yes | True | True | True | True | 24270957 / Is combined therapy more effective than growth hormone or hyperbaric oxygen alone in the healing of left ischemic and non-ischemic colonic anastomoses? |
| `pubmedqa-official-18507507` | yes | maybe | True | True | True | False | 18507507 / The promise of specialty pharmaceuticals: are they worth the price? |
| `pubmedqa-official-16772913` | yes | maybe | False | False | True | False | 26471488 / Does Mammographic Density have an Impact on the Margin Re-excision Rate After Breast-Conserving Surgery? |
| `pubmedqa-official-12172698` | yes | no | True | True | True | False | 12172698 / Is withdrawal-induced anxiety in alcoholism based on beta-endorphin deficiency? |
| `pubmedqa-official-26460153` | yes | maybe | True | True | True | False | 26460153 / Cardiac reoperations in octogenarians: Do they really benefit? |
| `pubmedqa-official-12419743` | yes | no | True | True | True | False | 12419743 / Is first-line single-agent mitoxantrone in the treatment of high-risk metastatic breast cancer patients as effective as combination chemotherapy? |
| `pubmedqa-official-25725704` | yes | yes | True | True | True | True | 25725704 / Can clinical supervision sustain our workforce in the current healthcare landscape? |
| `pubmedqa-official-25669733` | yes | yes | True | True | True | True | 25669733 / Can distal ureteral diameter predict reflux resolution after endoscopic injection? |
| `pubmedqa-official-24614789` | yes | yes | True | True | True | True | 24614789 / Is lumbar drainage of postoperative cerebrospinal fluid fistula after spine surgery effective? |
| `pubmedqa-official-24996865` | yes | maybe | True | True | True | False | 24996865 / Assessing joint line positions by means of the contralateral knee: a new approach for planning knee revision surgery? |
| `pubmedqa-official-18928979` | yes | yes | True | True | True | True | 18928979 / Can myometrial electrical activity identify patients in preterm labor? |
| `pubmedqa-official-25699562` | yes | yes | True | True | True | True | 25699562 / Does the Transmissible Liability Index (TLI) assessed in late childhood predict suicidal symptoms at young adulthood? |
| `pubmedqa-official-24577079` | no | maybe | True | True | True | False | 24577079 / Does strategy training reduce age-related deficits in working memory? |
| `pubmedqa-official-24669960` | no | no | True | True | True | True | 24669960 / Does the sex of acute stroke patients influence the effectiveness of rt-PA? |
| `pubmedqa-official-15502995` | no | no | True | True | True | True | 15502995 / Does the early adopter of drugs exist? |
| `pubmedqa-official-21214884` | no | no | True | True | True | True | 21214884 / Can 'high-risk' human papillomaviruses (HPVs) be detected in human breast milk? |
| `pubmedqa-official-24476003` | no | no | True | True | True | True | 24476003 / Is nasogastric decompression useful in prevention of leaks after laparoscopic sleeve gastrectomy? |
| `pubmedqa-official-22758782` | no | maybe | True | True | True | False | 22758782 / Regional anesthesia as compared with general anesthesia for surgery in geriatric patients with hip fracture: does it decrease morbidity, mortality, and health care costs? |
| `pubmedqa-official-14627582` | no | yes | True | True | True | False | 14627582 / Double reading of barium enemas: is it necessary? |
| `pubmedqa-official-24666444` | no | no | True | True | True | True | 24666444 / Is there any evidence of a "July effect" in patients undergoing major cancer surgery? |
| `pubmedqa-official-18496363` | no | yes | True | True | True | False | 18496363 / Characterization of the gender dimorphism after injury and hemorrhagic shock: are hormonal differences responsible? |
| `pubmedqa-official-12040336` | no | maybe | True | True | True | False | 12040336 / Cardiogenic shock complicating acute myocardial infarction in elderly patients: does admission to a tertiary center improve survival? |
| `pubmedqa-official-14631523` | no | no | True | True | True | True | 14631523 / Sub-classification of low-grade cerebellar astrocytoma: is it clinically meaningful? |
| `pubmedqa-official-21823940` | no | maybe | True | True | True | False | 21823940 / Department of Transportation vs self-reported data on motor vehicle collisions and driving convictions for stroke survivors: do they agree? |
| `pubmedqa-official-17971187` | no | no | True | True | True | True | 17971187 / Cholesterol screening in school children: is family history reliable to choose the ones to screen? |
| `pubmedqa-official-27642458` | no | maybe | True | True | True | False | 27642458 / Did the call for boycott by the Catholic bishops affect the polio vaccination coverage in Kenya in 2015? |
| `pubmedqa-official-12970636` | no | maybe | True | True | True | False | 12970636 / Does early discharge with nurse home visits affect adequacy of newborn metabolic screening? |
| `pubmedqa-official-11138995` | no | maybe | True | True | True | False | 11138995 / Is alexithymia a risk factor for unexplained physical symptoms in general medical outpatients? |
| `pubmedqa-official-15388567` | no | maybe | True | True | True | False | 15388567 / Are sports medicine journals relevant and applicable to practitioners and athletes? |
| `pubmedqa-official-19142546` | no | no | True | True | True | True | 19142546 / Does quantitative left ventricular regional wall motion change after fibrous tissue resection in endomyocardial fibrosis? |
| `pubmedqa-official-8921484` | no | maybe | True | True | True | False | 8921484 / Does gestational age misclassification explain the difference in birthweights for Australian aborigines and whites? |
| `pubmedqa-official-26209118` | no | no | True | True | True | True | 26209118 / Utility of unenhanced fat-suppressed T1-weighted MRI in children with sickle cell disease -- can it differentiate bone infarcts from acute osteomyelitis? |
| `pubmedqa-official-22668852` | no | no | True | True | True | True | 22668852 / Do African American women require fewer calories to maintain weight? |
| `pubmedqa-official-18019905` | no | maybe | True | True | True | False | 18019905 / The use of audit to identify maternal mortality in different settings: is it just a difference between the rich and the poor? |
| `pubmedqa-official-18378554` | no | maybe | True | True | True | False | 18378554 / Are wandering and physically nonaggressive agitation equivalent? |
| `pubmedqa-official-24073931` | no | no | True | True | True | True | 24073931 / Is the covering of the resection margin after distal pancreatectomy advantageous? |
| `pubmedqa-official-7547656` | no | no | True | True | True | True | 7547656 / Does continuous intravenous infusion of low-concentration epinephrine impair uterine blood flow in pregnant ewes? |
| `pubmedqa-official-28359277` | no | yes | True | True | True | False | 28359277 / Do healthier lifestyles lead to less utilization of healthcare resources? |
| `pubmedqa-official-18667100` | no | no | True | True | True | True | 18667100 / Do risk factors for suicidal behavior differ by affective disorder polarity? |
| `pubmedqa-official-10781708` | no | maybe | True | True | True | False | 10781708 / Thrombosis prophylaxis in hospitalised medical patients: does prophylaxis in all patients make sense? |
| `pubmedqa-official-22522271` | no | maybe | False | False | True | False | 26923375 / Is non-invasive diagnosis of esophageal varices in patients with compensated hepatic cirrhosis possible by duplex Doppler ultrasonography? |
| `pubmedqa-official-11955750` | no | yes | True | True | True | False | 11955750 / Does escalation of the apical dose change treatment outcome in beta-radiation of posterior choroidal melanomas with 106Ru plaques? |
| `pubmedqa-official-26126304` | no | no | True | True | True | True | 26126304 / Estradiol and Antagonist Pretreatment Prior to Microdose Leuprolide in in Vitro Fertilization. Does It Improve IVF Outcomes in Poor Responders as Compared to Oral Contraceptive Pill? |
| `pubmedqa-official-27338535` | no | maybe | True | True | True | False | 27338535 / Do risk calculators accurately predict surgical site occurrences? |
| `pubmedqa-official-24799031` | no | no | True | True | True | True | 24799031 / Is diffusion-weighted imaging a significant indicator of the development of vascularization in hypovascular hepatocellular lesions? |
| `pubmedqa-official-18319270` | no | maybe | True | True | True | False | 18319270 / Does confined placental mosaicism account for adverse perinatal outcomes in IVF pregnancies? |
| `pubmedqa-official-21789019` | no | maybe | True | True | True | False | 21789019 / Do elderly cancer patients have different care needs compared with younger ones? |
| `pubmedqa-official-11567820` | no | no | True | True | True | True | 11567820 / Does increased nerve length within the treatment volume improve trigeminal neuralgia radiosurgery? |
| `pubmedqa-official-10966943` | no | yes | True | True | True | False | 10966943 / Amblyopia: is visual loss permanent? |
| `pubmedqa-official-8199520` | no | no | True | True | True | True | 8199520 / Are physicians meeting the needs of family caregivers of the frail elderly? |
| `pubmedqa-official-21889895` | no | maybe | True | True | True | False | 21889895 / Will CT ordering practices change if we educate residents about the potential effects of radiation exposure? |
| `pubmedqa-official-26113007` | no | no | True | True | True | True | 26113007 / Is arch form influenced by sagittal molar relationship or Bolton tooth-size discrepancy? |
| `pubmedqa-official-17208539` | no | no | True | True | True | True | 17208539 / Are the long-term results of the transanal pull-through equal to those of the transabdominal pull-through? |
| `pubmedqa-official-20538207` | no | maybe | True | True | True | False | 20538207 / Should temperature be monitorized during kidney allograft preservation? |
| `pubmedqa-official-9603166` | no | no | True | True | True | True | 9603166 / Should all human immunodeficiency virus-infected patients with end-stage renal disease be excluded from transplantation? |
| `pubmedqa-official-21194998` | no | no | True | True | True | True | 21194998 / Does minimal access major surgery in the newborn hurt less? |
| `pubmedqa-official-21252642` | no | maybe | True | True | True | False | 21252642 / Does increased patient awareness improve accrual into cancer-related clinical trials? |
| `pubmedqa-official-16678696` | no | maybe | True | True | True | False | 16678696 / Continuity of care experience of residents in an academic vascular department: are trainees learning complete surgical care? |
| `pubmedqa-official-20549895` | no | maybe | True | True | True | False | 20549895 / Health habits and vaccination status of Lebanese residents: are future doctors applying the rules of prevention? |
| `pubmedqa-official-16418930` | no | maybe | True | True | True | False | 16418930 / Landolt C and snellen e acuity: differences in strabismus amblyopia? |
| `pubmedqa-official-8521557` | no | maybe | True | True | True | False | 8521557 / The insertion allele of the ACE gene I/D polymorphism. A candidate gene for insulin resistance? |
| `pubmedqa-official-16809243` | no | no | True | True | True | True | 16809243 / Is fetal gender associated with emergency department visits for asthma during pregnancy? |
| `pubmedqa-official-10798511` | no | maybe | True | True | True | False | 10798511 / Blunt trauma in intoxicated patients: is computed tomography of the abdomen always necessary? |
| `pubmedqa-official-10834864` | no | no | True | True | True | True | 10834864 / Risk factors for avascular necrosis of bone in patients with systemic lupus erythematosus: is there a role for antiphospholipid antibodies? |
| `pubmedqa-official-16962519` | no | maybe | True | True | True | False | 16962519 / Volume change of uterine myomas during pregnancy: do myomas really grow? |
| `pubmedqa-official-19575104` | no | yes | True | True | True | False | 19575104 / Do foreign bodies migrate through the body towards the heart? |
| `pubmedqa-official-24809662` | no | no | True | True | True | True | 24809662 / Does concomitant anterior/apical repair during midurethral sling improve the overactive bladder component of mixed incontinence? |
| `pubmedqa-official-20602101` | no | no | True | True | True | True | 20602101 / Is hypoalbuminemia an independent prognostic factor in patients with gastric cancer? |
| `pubmedqa-official-26852225` | no | yes | True | True | True | False | 26852225 / Is adjustment for reporting heterogeneity necessary in sleep disorders? |
| `pubmedqa-official-19398929` | no | no | True | True | True | True | 19398929 / Can the growth rate of a gallbladder polyp predict a neoplastic polyp? |
| `pubmedqa-official-25614468` | no | maybe | True | True | True | False | 25614468 / Preoperative locoregional staging of gastric cancer: is there a place for magnetic resonance imaging? |
| `pubmedqa-official-11926574` | no | maybe | True | True | True | False | 11926574 / Are hepatitis G virus and TT virus involved in cryptogenic chronic liver disease? |
| `pubmedqa-official-10973547` | no | maybe | True | True | True | False | 10973547 / Are patients with Werlhof's disease at increased risk for bleeding complications when undergoing cardiac surgery? |
| `pubmedqa-official-26471488` | no | no | True | True | True | True | 26471488 / Does Mammographic Density have an Impact on the Margin Re-excision Rate After Breast-Conserving Surgery? |
| `pubmedqa-official-19520213` | no | maybe | True | True | True | False | 19520213 / Are UK radiologists satisfied with the training and support received in suspected child abuse? |
| `pubmedqa-official-23677366` | no | no | True | True | True | True | 23677366 / Do oblique views add value in the diagnosis of spondylolysis in adolescents? |
| `pubmedqa-official-17342562` | no | no | True | True | True | True | 17342562 / The clinical significance of bile duct sludge: is it different from bile duct stones? |
| `pubmedqa-official-16296668` | no | maybe | True | True | True | False | 16296668 / Can bedside assessment reliably exclude aspiration following acute stroke? |
| `pubmedqa-official-17054994` | no | no | True | True | True | True | 17054994 / Does frozen section alter surgical management of multinodular thyroid disease? |
| `pubmedqa-official-26556589` | no | no | True | True | True | True | 26556589 / Does type 1 diabetes mellitus affect Achilles tendon response to a 10 km run? |
| `pubmedqa-official-15052394` | no | no | True | True | True | True | 15052394 / Are higher rates of depression in women accounted for by differential symptom reporting? |
| `pubmedqa-official-22513023` | no | yes | True | True | True | False | 22513023 / Do Indigenous Australians age prematurely? |
| `pubmedqa-official-15919266` | no | no | True | True | True | True | 15919266 / Adjuvant radiation of stage III thymoma: is it necessary? |
| `pubmedqa-official-15095519` | no | no | True | True | True | True | 15095519 / Are patients with diabetes receiving the same message from dietitians and nurses? |
| `pubmedqa-official-12006913` | no | maybe | True | True | True | False | 12006913 / Do lipids, blood pressure, diabetes, and smoking confer equal risk of myocardial infarction in women as in men? |
| `pubmedqa-official-8738894` | no | yes | True | True | True | False | 8738894 / Diabetes mellitus among Swedish art glass workers--an effect of arsenic exposure? |
| `pubmedqa-official-21431987` | no | maybe | True | True | True | False | 21431987 / Preoperative staging of patients with liver metastases of colorectal carcinoma. Does PET/CT really add something to multidetector CT? |
| `pubmedqa-official-22154448` | no | no | True | True | True | True | 22154448 / Epidural analgesia for surgical treatment of peritoneal carcinomatosis: a risky technique? |
| `pubmedqa-official-15053041` | no | no | True | True | True | True | 15053041 / Do acute changes in heart rate by isoproterenol affect aortic stiffness in patients with hypertension? |
| `pubmedqa-official-22365295` | no | maybe | True | True | True | False | 22365295 / Totally implantable venous access device placement by interventional radiologists: are prophylactic antibiotics necessary? |
| `pubmedqa-official-19546588` | no | no | True | True | True | True | 19546588 / Does increasing blood pH stimulate protein synthesis in dialysis patients? |
| `pubmedqa-official-7482275` | no | maybe | True | True | True | False | 7482275 / Necrotizing fasciitis: an indication for hyperbaric oxygenation therapy? |
| `pubmedqa-official-24698298` | no | no | True | True | True | True | 24698298 / MR arthrography of the shoulder: do we need local anesthesia? |
| `pubmedqa-official-18274917` | no | no | True | True | True | True | 18274917 / Prognosis of low-tone sudden deafness - does it inevitably progress to Meniere's disease? |
| `pubmedqa-official-21946341` | no | no | True | True | True | True | 21946341 / Is there a relationship between complex fractionated atrial electrograms recorded during atrial fibrillation and sinus rhythm fractionation? |
| `pubmedqa-official-23568387` | no | no | True | True | True | True | 23568387 / Is bicompartmental knee arthroplasty more favourable to knee muscle strength and physical performance compared to total knee arthroplasty? |
| `pubmedqa-official-21256734` | no | no | True | True | True | True | 21256734 / Does pain intensity predict a poor opioid response in cancer patients? |
| `pubmedqa-official-22534881` | no | maybe | True | True | True | False | 22534881 / Does the radiographic transition zone correlate with the level of aganglionosis on the specimen in Hirschsprung's disease? |
| `pubmedqa-official-8566975` | no | no | True | True | True | True | 8566975 / Serovar specific immunity to Neisseria gonorrhoeae: does it exist? |
| `pubmedqa-official-23761381` | no | no | True | True | True | True | 23761381 / Is calibration the cause of variation in liquid chromatography tandem mass spectrometry testosterone measurement? |
| `pubmedqa-official-22668712` | no | no | True | True | True | True | 22668712 / Internal derangement of the temporomandibular joint: is there still a place for ultrasound? |
| `pubmedqa-official-22023714` | no | no | True | True | True | True | 22023714 / Does delivery mode affect women's postpartum quality of life in rural China? |
| `pubmedqa-official-22504515` | no | no | True | True | True | True | 22504515 / Endovenous laser ablation in the treatment of small saphenous varicose veins: does site of access influence early outcomes? |
| `pubmedqa-official-21164063` | no | maybe | True | True | True | False | 21164063 / Is there a role for fondaparinux in perioperative bridging? |
| `pubmedqa-official-18359123` | no | no | False | False | True | False | 26516021 / Does evidence-based practice improve patient outcomes? |
| `pubmedqa-official-16827975` | no | yes | True | True | True | False | 16827975 / Chemotherapy and survival in advanced non-small cell lung carcinoma: is pneumologists' skepticism justified? |
| `pubmedqa-official-24922528` | no | maybe | True | True | True | False | 24922528 / The association of puberty and young adolescent alcohol use: do parents have a moderating role? |
| `pubmedqa-official-15774570` | no | maybe | True | True | True | False | 15774570 / Does increased use of private health care reduce the demand for NHS care? |
| `pubmedqa-official-20736887` | no | maybe | True | True | True | False | 20736887 / Is decompressive surgery effective for spinal cord sarcoidosis accompanied with compressive cervical myelopathy? |
| `pubmedqa-official-11483547` | no | yes | True | True | True | False | 11483547 / Does the aggressive use of polyvalent antivenin for rattlesnake bites result in serious acute side effects? |
| `pubmedqa-official-9542484` | no | no | True | True | True | True | 9542484 / Does successful completion of the Perinatal Education Programme result in improved obstetric practice? |
| `pubmedqa-official-18708308` | no | no | True | True | True | True | 18708308 / Can surgeon familiarization with current evidence lead to a change in practice? |
| `pubmedqa-official-18435678` | no | no | True | True | True | True | 18435678 / Kell alloimmunization in pregnancy: associated with fetal thrombocytopenia? |
| `pubmedqa-official-23455575` | no | maybe | True | True | True | False | 23455575 / Globulomaxillary cysts--do they really exist? |
| `pubmedqa-official-22537902` | no | maybe | True | True | True | False | 22537902 / Colorectal cancer with synchronous liver metastases: does global management at the same centre improve results? |
| `pubmedqa-official-18926458` | no | yes | True | True | True | False | 18926458 / Are octogenarians at high risk for carotid endarterectomy? |
| `pubmedqa-official-12090319` | no | no | True | True | True | True | 12090319 / Is there a need for pelvic CT scan in cases of renal cell carcinoma? |
| `pubmedqa-official-12380309` | no | maybe | True | True | True | False | 12380309 / Should circumcision be performed in childhood? |
| `pubmedqa-official-27989969` | no | no | True | True | True | True | 27989969 / Does the Simultaneous Use of a Neuroendoscope Influence the Incidence of Ventriculoperitoneal Shunt Infection? |
| `pubmedqa-official-25752912` | no | no | True | True | True | True | 25752912 / Is the probability of prenatal diagnosis or termination of pregnancy different for fetuses with congenital anomalies conceived following assisted reproductive techniques? |
| `pubmedqa-official-26536001` | no | no | True | True | True | True | 26536001 / Is There an Additional Value of Using Somatostatin Receptor Subtype 2a Immunohistochemistry Compared to Somatostatin Receptor Scintigraphy Uptake in Predicting Gastroenteropancreatic Neuroendocrine Tumor Response? |
| `pubmedqa-official-21849531` | no | maybe | True | True | True | False | 21849531 / Does growth hormone replacement therapy reduce mortality in adults with growth hormone deficiency? |
| `pubmedqa-official-16872243` | no | yes | True | True | True | False | 16872243 / Can decisional algorithms replace global introspection in the individual causality assessment of spontaneously reported ADRs? |
| `pubmedqa-official-23571528` | no | maybe | True | True | True | False | 23571528 / Sternal skin conductance: a reasonable surrogate for hot flash measurement? |
| `pubmedqa-official-19481382` | no | maybe | True | True | True | False | 19481382 / Is the Androgen Deficiency of Aging Men (ADAM) questionnaire useful for the screening of partial androgenic deficiency of aging men? |
| `pubmedqa-official-23621776` | no | maybe | True | True | True | False | 23621776 / Does a history of unintended pregnancy lessen the likelihood of desire for sterilization reversal? |
| `pubmedqa-official-22227642` | no | maybe | True | True | True | False | 22227642 / Can we measure mesopic pupil size with the cobalt blue light slit-lamp biomicroscopy method? |
| `pubmedqa-official-23025584` | no | no | True | True | True | True | 23025584 / Does stress increase imitation of drinking behavior? |
| `pubmedqa-official-11862129` | no | maybe | True | True | True | False | 11862129 / Do clinical variables predict pathologic radiographs in the first episode of wheezing? |
| `pubmedqa-official-22236315` | no | yes | True | True | True | False | 22236315 / Is distance to provider a barrier to care for medicaid patients with breast, colorectal, or lung cancer? |
| `pubmedqa-official-21361755` | no | no | True | True | True | True | 21361755 / Laminoplasty outcomes: is there a difference between patients with degenerative stenosis and those with ossification of the posterior longitudinal ligament? |
| `pubmedqa-official-18719011` | no | maybe | True | True | True | False | 18719011 / Do overweight children necessarily make overweight adults? |
| `pubmedqa-official-11438275` | no | no | True | True | True | True | 11438275 / Does patient position during liver surgery influence the risk of venous air embolism? |
| `pubmedqa-official-16778275` | no | maybe | True | True | True | False | 16778275 / Is routine chest radiography after transbronchial biopsy necessary? |
| `pubmedqa-official-17051586` | no | no | True | True | True | True | 17051586 / Can folic acid protect against congenital heart defects in Down syndrome? |
| `pubmedqa-official-24061619` | no | no | True | True | True | True | 24061619 / Location and number of sutures placed for hiatal hernia repair during laparoscopic adjustable gastric banding: does it matter? |
| `pubmedqa-official-22233470` | no | no | True | True | True | True | 22233470 / Does the distribution of health care benefits in Kenya meet the principles of universal coverage? |
| `pubmedqa-official-23497210` | no | maybe | True | True | True | False | 23497210 / Are women with major depression in pregnancy identifiable in population health data? |
| `pubmedqa-official-25488308` | no | no | True | True | True | True | 25488308 / Can bone thickness and inter-radicular space affect miniscrew placement in posterior mandibular sites? |
| `pubmedqa-official-22382608` | no | maybe | True | True | True | False | 22382608 / SPECT study with I-123-Ioflupane (DaTSCAN) in patients with essential tremor. Is there any correlation with Parkinson's disease? |
| `pubmedqa-official-19237087` | no | no | True | True | True | True | 19237087 / Are many colorectal cancers due to missed adenomas? |
| `pubmedqa-official-10381996` | no | no | True | True | True | True | 10381996 / Clinician assessment for acute chest syndrome in febrile patients with sickle cell disease: is it accurate enough? |
| `pubmedqa-official-9100537` | no | no | True | True | True | True | 9100537 / Can nonproliferative breast disease and proliferative breast disease without atypia be distinguished by fine-needle aspiration cytology? |
| `pubmedqa-official-23422012` | no | maybe | True | True | True | False | 23422012 / Is vancomycin MIC creep a worldwide phenomenon? |
| `pubmedqa-official-22876568` | no | no | True | True | True | True | 22876568 / Is vitamin D deficiency a feature of pediatric celiac disease? |
| `pubmedqa-official-17445978` | no | no | True | True | True | True | 17445978 / Is renal warm ischemia over 30 minutes during laparoscopic partial nephrectomy possible? |
| `pubmedqa-official-20608141` | no | no | True | True | True | True | 20608141 / PSA repeatedly fluctuating levels are reassuring enough to avoid biopsy? |
| `pubmedqa-official-23177368` | no | no | True | True | True | True | 23177368 / Does immediate breast reconstruction compromise the delivery of adjuvant chemotherapy? |
| `pubmedqa-official-8847047` | no | no | True | True | True | True | 8847047 / Prognosis of well differentiated small hepatocellular carcinoma--is well differentiated hepatocellular carcinoma clinically early cancer? |
| `pubmedqa-official-22011946` | no | no | True | True | True | True | 22011946 / Does a preoperative medically supervised weight loss program improve bariatric surgery outcomes? |
| `pubmedqa-official-27394685` | no | no | True | True | True | True | 27394685 / Bony defects in chronic anterior posttraumatic dislocation of the shoulder: Is there a correlation between humeral and glenoidal lesions? |
| `pubmedqa-official-23794696` | no | no | True | True | True | True | 23794696 / Does the bracket-ligature combination affect the amount of orthodontic space closure over three months? |
| `pubmedqa-official-23076787` | no | maybe | True | True | True | False | 23076787 / Can increases in the cigarette tax rate be linked to cigarette retail prices? |
| `pubmedqa-official-19854401` | no | maybe | True | True | True | False | 19854401 / Attaining negative margins in breast-conservation operations: is there a consensus among breast surgeons? |
| `pubmedqa-official-14652839` | no | no | True | True | True | True | 14652839 / Does the sequence of clamp application during open abdominal aortic aneurysm surgery influence distal embolisation? |
| `pubmedqa-official-7664228` | no | maybe | True | True | True | False | 7664228 / Discharging patients earlier from Winnipeg hospitals: does it adversely affect quality of care? |
| `pubmedqa-official-17062234` | no | yes | True | True | True | False | 17062234 / Surgical management of the atherosclerotic ascending aorta: is endoaortic balloon occlusion safe? |
| `pubmedqa-official-24449622` | no | maybe | True | True | True | False | 24449622 / Is there a relationship between serum paraoxonase level and epicardial fat tissue thickness? |
| `pubmedqa-official-12070552` | no | no | True | True | True | True | 12070552 / Do antibiotics decrease post-tonsillectomy morbidity? |
| `pubmedqa-official-19836806` | no | no | True | True | True | True | 19836806 / Should prostate specific antigen be adjusted for body mass index? |
| `pubmedqa-official-19913785` | no | no | True | True | True | True | 19913785 / Is it necessary to insert a nasobiliary drainage tube routinely after endoscopic clearance of the common bile duct in patients with choledocholithiasis-induced cholangitis? |
| `pubmedqa-official-24739448` | no | no | True | True | True | True | 24739448 / Have antiepileptic drug prescription claims changed following the FDA suicidality warning? |
| `pubmedqa-official-24625433` | no | no | True | True | True | True | 24625433 / Are high flow nasal cannulae noisier than bubble CPAP for preterm infants? |
| `pubmedqa-official-16403186` | no | no | True | True | True | True | 16403186 / Are the arginine vasopressin V1a receptor microsatellites related to hypersexuality in children with a prepubertal and early adolescent bipolar disorder phenotype? |
| `pubmedqa-official-10375486` | no | no | True | True | True | True | 10375486 / Are variations in the use of carotid endarterectomy explained by population Need? |
| `pubmedqa-official-17032327` | no | no | True | True | True | True | 17032327 / Do supervised colorectal trainees differ from consultants in terms of quality of TME surgery? |
| `pubmedqa-official-27643961` | no | maybe | True | True | True | False | 27643961 / Major depression and alcohol use disorder in adolescence: Does comorbidity lead to poorer outcomes of depression? |
| `pubmedqa-official-22042121` | no | no | True | True | True | True | 22042121 / Perioperative care in an animal model for training in abdominal surgery: is it necessary a preoperative fasting? |
| `pubmedqa-official-17274051` | no | no | True | True | True | True | 17274051 / Metastatic carcinoma to the cervical nodes from an unknown head and neck primary site: Is there a need for neck dissection? |
| `pubmedqa-official-27096199` | no | maybe | True | True | True | False | 27096199 / Does Viral Co-Infection Influence the Severity of Acute Respiratory Infection in Children? |
| `pubmedqa-official-7497757` | no | yes | True | True | True | False | 7497757 / Cardiopulmonary bypass temperature does not affect postoperative euthyroid sick syndrome? |
| `pubmedqa-official-21459725` | no | no | True | True | True | True | 21459725 / Xanthogranulomatous cholecystitis: a premalignant condition? |
| `pubmedqa-official-27040842` | no | no | True | True | True | True | 27040842 / Does septoplasty change the dimensions of compensatory hypertrophy of the middle turbinate? |
| `pubmedqa-official-20187289` | no | maybe | True | True | True | False | 20187289 / Prescriptions as a proxy for asthma in children: a good choice? |
| `pubmedqa-official-21712147` | no | no | True | True | True | True | 21712147 / Does combining antiretroviral agents in a single dosage form enhance quality of life of HIV/AIDS patients? |
| `pubmedqa-official-10456814` | no | yes | True | True | True | False | 10456814 / Does desflurane alter left ventricular function when used to control surgical stimulation during aortic surgery? |
| `pubmedqa-official-17192736` | no | yes | True | True | True | False | 17192736 / Is fluoroscopy essential for retrieval of lower ureteric stones? |
| `pubmedqa-official-27757987` | no | no | True | True | True | True | 27757987 / Does the treatment of amblyopia normalise subfoveal choroidal thickness in amblyopic children? |
| `pubmedqa-official-12769830` | no | maybe | True | True | True | False | 12769830 / Should tumor depth be included in prognostication of soft tissue sarcoma? |
| `pubmedqa-official-22251324` | no | no | True | True | True | True | 22251324 / Does performance in selection processes predict performance as a dental student? |
| `pubmedqa-official-28196511` | no | maybe | True | True | True | False | 28196511 / Antiretroviral therapy related adverse effects: Can sub-Saharan Africa cope with the new "test and treat" policy of the World Health Organization? |
| `pubmedqa-official-18284441` | maybe | yes | True | True | True | False | 18284441 / Expression of c-kit protooncogen in hepatitis B virus-induced chronic hepatitis, cirrhosis and hepatocellular carcinoma: has it a diagnostic role? |
| `pubmedqa-official-18802997` | maybe | yes | True | True | True | False | 18802997 / Can calprotectin predict relapse risk in inflammatory bowel disease? |
| `pubmedqa-official-17621202` | maybe | no | True | True | True | False | 17621202 / Does shaving the incision site increase the infection rate after spinal surgery? |
| `pubmedqa-official-11411430` | maybe | yes | True | True | True | False | 11411430 / Antral follicle assessment as a tool for predicting outcome in IVF--is it a better predictor than age and FSH? |
| `pubmedqa-official-26708803` | maybe | maybe | True | True | True | True | 26708803 / Treatment of contralateral hydrocele in neonatal testicular torsion: Is less more? |
| `pubmedqa-official-25079920` | maybe | maybe | True | True | True | True | 25079920 / Do parents recall and understand children's weight status information after BMI screening? |
| `pubmedqa-official-25793749` | maybe | maybe | True | True | True | True | 25793749 / Do Web-based and clinic samples of gay men living with HIV differ on self-reported physical and psychological symptoms? |
| `pubmedqa-official-19103915` | maybe | maybe | True | True | True | True | 19103915 / Are home sampling kits for sexually transmitted infections acceptable among men who have sex with men? |
| `pubmedqa-official-11867487` | maybe | maybe | True | True | True | True | 11867487 / Does rugby headgear prevent concussion? |
| `pubmedqa-official-12630042` | maybe | yes | True | True | True | False | 12630042 / Does body mass index (BMI) influence morbidity and long-term survival in gastric cancer patients after gastrectomy? |
| `pubmedqa-official-19468282` | maybe | yes | True | True | True | False | 19468282 / Is determination between complete and incomplete traumatic spinal cord injury clinically relevant? |
| `pubmedqa-official-16538201` | maybe | yes | True | True | True | False | 16538201 / Does use of hydrophilic guidewires significantly improve technical success rates of peripheral PTA? |
| `pubmedqa-official-20971618` | maybe | maybe | True | True | True | True | 20971618 / Are lifetime prevalence of impetigo, molluscum and herpes infection really increased in children having atopic dermatitis? |
| `pubmedqa-official-24336869` | maybe | maybe | True | True | True | True | 24336869 / Can routinely collected ambulance data about assaults contribute to reduction in community violence? |
| `pubmedqa-official-20197761` | maybe | no | True | True | True | False | 20197761 / Is irritable bowel syndrome a diagnosis of exclusion? |
| `pubmedqa-official-16968876` | maybe | yes | True | True | True | False | 16968876 / Is a patient's self-reported health-related quality of life a prognostic factor for survival in non-small-cell lung cancer patients? |
| `pubmedqa-official-26778755` | maybe | no | True | True | True | False | 26778755 / Vaginal dose assessment in image-guided brachytherapy for cervical cancer: Can we really rely on dose-point evaluation? |
| `pubmedqa-official-18568290` | maybe | maybe | True | True | True | True | 18568290 / Is there a role for endothelin-1 in the hemodynamic changes during hemodialysis? |
| `pubmedqa-official-25371231` | maybe | maybe | True | True | True | True | 25371231 / Is vitamin D insufficiency or deficiency related to the development of osteochondritis dissecans? |
| `pubmedqa-official-25394614` | maybe | maybe | True | True | True | True | 25394614 / Does timing of initial surfactant treatment make a difference in rates of chronic lung disease or mortality in premature infants? |
| `pubmedqa-official-11570976` | maybe | maybe | False | False | True | False | 10223070 / Is perforation of the appendix a risk factor for tubal infertility and ectopic pregnancy? |
| `pubmedqa-official-16816043` | maybe | yes | True | True | True | False | 16816043 / Do French lay people and health professionals find it acceptable to breach confidentiality to protect a patient's wife from a sexually transmitted disease? |
| `pubmedqa-official-12805495` | maybe | maybe | True | True | True | True | 12805495 / Can patients be anticoagulated after intracerebral hemorrhage? |
| `pubmedqa-official-25571931` | maybe | maybe | True | True | True | True | 25571931 / Do elderly patients call 911 when presented with clinical scenarios suggestive of acute stroke? |
| `pubmedqa-official-19578820` | maybe | maybe | True | True | True | True | 19578820 / Are opioid dependence and methadone maintenance treatment (MMT) documented in the medical record? |
| `pubmedqa-official-18243752` | maybe | maybe | True | True | True | True | 18243752 / Should chest wall irradiation be included after mastectomy and negative node breast cancer? |
| `pubmedqa-official-11458136` | maybe | yes | True | True | True | False | 11458136 / Does managed care enable more low income persons to identify a usual source of care? |
| `pubmedqa-official-12790890` | maybe | maybe | True | True | True | True | 12790890 / Is the cell death in mesial temporal sclerosis apoptotic? |
| `pubmedqa-official-18714572` | maybe | no | True | True | True | False | 18714572 / Does vaginal intraepithelial neoplasia have the same evolution as cervical intraepithelial neoplasia? |
| `pubmedqa-official-25103647` | maybe | maybe | True | True | True | True | 25103647 / Does government assistance improve utilization of eye care services by low-income individuals? |
| `pubmedqa-official-24995509` | maybe | no | True | True | True | False | 24995509 / HIF1A as a major vascular endothelial growth factor regulator: do its polymorphisms have an association with age-related macular degeneration? |
| `pubmedqa-official-27044366` | maybe | no | True | True | True | False | 27044366 / Detailed analysis of sputum and systemic inflammation in asthma phenotypes: are paucigranulocytic asthmatics really non-inflammatory? |
| `pubmedqa-official-26606599` | maybe | yes | True | True | True | False | 26606599 / Do Surrogates of Injury Severity Influence the Occurrence of Heterotopic Ossification in Fractures of the Acetabulum? |
| `pubmedqa-official-17076091` | maybe | maybe | True | True | True | False | 17076091 / Does obstructive sleep apnea affect aerobic fitness? |
| `pubmedqa-official-26037986` | maybe | yes | True | True | True | False | 26037986 / 30-Day and 1-year mortality in emergency general surgery laparotomies: an area of concern and need for improvement? |
| `pubmedqa-official-22491528` | maybe | yes | True | True | True | False | 22491528 / Combining process indicators to evaluate quality of care for surgical patients with colorectal cancer: are scores consistent with short-term outcome? |
| `pubmedqa-official-24591144` | maybe | maybe | True | True | True | True | 24591144 / Are the elderly with oropharyngeal carcinoma undertreated? |
| `pubmedqa-official-19351635` | maybe | maybe | True | True | True | True | 19351635 / Do older patients receive adequate stroke care? |
| `pubmedqa-official-20337202` | maybe | maybe | True | True | True | True | 20337202 / Continuation of pregnancy after antenatal corticosteroid administration: opportunity for rescue? |
| `pubmedqa-official-23149821` | maybe | maybe | True | True | True | True | 23149821 / Should HIV-infected patients be screened for silent myocardial ischaemia using gated myocardial perfusion SPECT? |
| `pubmedqa-official-18235194` | maybe | maybe | True | True | True | True | 18235194 / Is a specialised training of phonological awareness indicated in every preschool child? |
| `pubmedqa-official-16392897` | maybe | maybe | True | True | True | True | 16392897 / BCRABL transcript detection by quantitative real-time PCR : are correlated results possible from homebrew assays? |
| `pubmedqa-official-17940352` | maybe | yes | True | True | True | False | 17940352 / Does HER2 immunoreactivity provide prognostic information in locally advanced urothelial carcinoma patients receiving adjuvant M-VEC chemotherapy? |
| `pubmedqa-official-27615402` | maybe | no | True | True | True | False | 27615402 / Does the familial transmission of drinking patterns persist into young adulthood? |
| `pubmedqa-official-25779009` | maybe | maybe | False | False | True | False | 27287237 / Assessment of appropriate antimicrobial prescribing: do experts agree? |
| `pubmedqa-official-12407608` | maybe | yes | True | True | True | False | 12407608 / Does ultrasound imaging before puncture facilitate internal jugular vein cannulation? |
| `pubmedqa-official-14599616` | maybe | maybe | True | True | True | False | 14599616 / Can a practicing surgeon detect early lymphedema reliably? |
| `pubmedqa-official-10223070` | maybe | maybe | True | True | True | True | 10223070 / Is perforation of the appendix a risk factor for tubal infertility and ectopic pregnancy? |
| `pubmedqa-official-20736672` | maybe | yes | True | True | True | False | 20736672 / Does perspective-taking increase patient satisfaction in medical encounters? |
| `pubmedqa-official-25277731` | maybe | yes | True | True | True | False | 25277731 / Sternal fracture in growing children : A rare and often overlooked fracture? |
| `pubmedqa-official-17691856` | maybe | maybe | True | True | True | True | 17691856 / Midwives' competence: is it affected by working in a rural location? |
| `pubmedqa-official-16735905` | maybe | yes | True | True | True | False | 16735905 / Does the severity of obstructive sleep apnea predict patients requiring high continuous positive airway pressure? |
| `pubmedqa-official-19694846` | maybe | no | True | True | True | False | 19694846 / Does self-efficacy mediate the relationship between transformational leadership behaviours and healthcare workers' sleep quality? |
| `pubmedqa-official-25007420` | maybe | maybe | True | True | True | True | 25007420 / Are there mental health differences between francophone and non-francophone populations in manitoba? |
| `pubmedqa-official-26134053` | maybe | maybe | True | True | True | True | 26134053 / Outcome Feedback within Emergency Medicine Training Programs: An Opportunity to Apply the Theory of Deliberate Practice? |

## Config

- `candidate_k`: `20`
- `top_k`: `1`
- `temperature`: `0.0`
- `RAG_EVIDENCE_FILTER_ENABLED`: `true`
- `RAG_ANSWER_QUALITY_GATE_ENABLED`: `true`
- `RAG_CORPUS_VERSION`: `pubmedqa-official-pqal-test-v1`
