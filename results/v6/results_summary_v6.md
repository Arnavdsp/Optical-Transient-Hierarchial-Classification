## v6 results

Objects: 1681 SNe (SN_Ia 600, SN_II 600, SLSN 202, SN_Ic 183, SN_Ib 96), against 150 in v5.

**5-way**: CV accuracy 0.607 (v5 setup) -> 0.821 (v6); held-out test with HistGB + SMOTE: accuracy 0.804 (95 % CI 0.757-0.846), balanced accuracy 0.657, macro-F1 0.652. With the 90% confidence cut fixed on training data: test accuracy 0.903 on 79% of test objects.

**4-way**: CV accuracy 0.773 (v5 setup) -> 0.860 (v6); held-out test with Soft vote HGB+RF+ET: accuracy 0.861 (95 % CI 0.822-0.896), balanced accuracy 0.847, macro-F1 0.841. With the 90% confidence cut fixed on training data: test accuracy 0.913 on 85% of test objects.

**4-way SESN**: held-out test with Soft vote HGB+RF+ET: accuracy 0.846 (95 % CI 0.804-0.884), balanced accuracy 0.839, macro-F1 0.833. With the 90% confidence cut fixed on training data: test accuracy 0.900 on 83% of test objects.

**3-way**: CV accuracy 0.917 (v5 setup) -> 0.927 (v6); held-out test with Soft vote HGB+RF+ET: accuracy 0.929 (95 % CI 0.899-0.955), balanced accuracy 0.915, macro-F1 0.918. With the 90% confidence cut fixed on training data: test accuracy 0.929 on 100% of test objects.

Caveat on redshift: redshift and absolute magnitude are legitimate physical features, but part of their power is the selection function of spectroscopic follow-up (SLSNe are found far away, Ib/c nearby). v5 STEP 11 measured this; it applies to any classifier trained on TNS-classified objects, including ALeRCE and Superphot+.
