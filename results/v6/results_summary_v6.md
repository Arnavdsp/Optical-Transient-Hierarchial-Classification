## v6 results

Objects: 1656 SNe (SN_Ia 600, SN_II 600, SLSN 185, SN_Ic 179, SN_Ib 92), against 150 in v5.

**5-way**: CV accuracy 0.600 (v5 setup) -> 0.823 (v6); held-out test with HistGB + SMOTE: accuracy 0.819 (95 % CI 0.780-0.858), balanced accuracy 0.713, macro-F1 0.697; on the 80 % most confident test objects: accuracy 0.929.

**4-way**: CV accuracy 0.770 (v5 setup) -> 0.855 (v6); held-out test with HistGB + SMOTE: accuracy 0.831 (95 % CI 0.789-0.873), balanced accuracy 0.808, macro-F1 0.802; on the 80 % most confident test objects: accuracy 0.902.

**4-way SESN**: held-out test with Hierarchical (HGB nodes): accuracy 0.828 (95 % CI 0.786-0.867), balanced accuracy 0.825, macro-F1 0.812; on the 80 % most confident test objects: accuracy 0.876.

**3-way**: CV accuracy 0.903 (v5 setup) -> 0.932 (v6); held-out test with Soft vote HGB+RF+ET: accuracy 0.946 (95 % CI 0.919-0.967), balanced accuracy 0.934, macro-F1 0.935; on the 80 % most confident test objects: accuracy 0.992.

Caveat on redshift: redshift and absolute magnitude are legitimate physical features, but part of their power is the selection function of spectroscopic follow-up (SLSNe are found far away, Ib/c nearby). v5 STEP 11 measured this; it applies to any classifier trained on TNS-classified objects, including ALeRCE and Superphot+.
