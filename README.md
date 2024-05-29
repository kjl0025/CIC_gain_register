# CIC_gain_register
EMCCD frame fitter that accounts for clock-induced charge (CIC) in the gain register

This project verifies experimentally a probability distribution function which models usual sources of noise for an electron-multiplying charge-coupled device (EMCCD) as well as a source of noise not usually taken into account, clock-induced charge in the gain register (what we call "partial CIC").  The script `partial_CIC_MLE.py` performs maximum likelihood estimation (MLE) over EMCCD data and finds the best-fit parameters characterizing the data.  The script shows that the inclusion of partial CIC provides a better statistical fit to EMCCD data, which results in more accurate estimation of EM gain applied to a frame.

Sample raw EMCCD data are also included.  The data are analyzed in this paper:
http://arxiv.org/abs/2405.17622

To process this sample EMCCD data, the script uses a package with simple installation instructions.  It can be found here:
https://github.com/roman-corgi/cgi_iit_drp/tree/main/Calibration_NTR

With this package installed, the script should run the sample data if it is located in a folder called `data_dir` in the same location as the script.
