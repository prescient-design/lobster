from collections.abc import Mapping
import math
from types import MappingProxyType
from typing import Protocol

import numpy as np
import scipy.stats
from numpy.typing import ArrayLike


# NOTE: This is a hack to get the CDF of the distribution. In the future, should use the new Distribution class (via
# `scipy.stats.make_distribution`) instead
class SupportsCDF(Protocol):
    def cdf(self, x: float | ArrayLike) -> float | np.ndarray: ...


DESCRIPTOR_DISTRIBUTIONS = {
    "VSA_EState1": (
        scipy.stats.betaprime(0.129794, 2.00845, -1.78743e-34, 0.402201),
        (0, 0, 0, 0),
    ),
    "Chi4n": (
        scipy.stats.mielke(3.43345, 4.64433, -0.154077, 3.77246),
        (0, 60.1557, 3.55837, 1.934),
    ),
    "EState_VSA3": (
        scipy.stats.pearson3(1.21303, 20.4904, 14.9136),
        (0, 707.419, 20.4905, 15.0269),
    ),
    "PEOE_VSA3": (
        scipy.stats.recipinvgauss(2.03999e6, -1.51416e-12, 5.86277),
        (0, 278.496, 5.88362, 7.11435),
    ),
    "PEOE_VSA10": (
        scipy.stats.ncx2(1.2635, 2.15031, -2.21123e-31, 2.60641),
        (0, 494.056, 9.76362, 12.9153),
    ),
    "Chi2v": (
        scipy.stats.fisk(5.41629, -0.467117, 7.91173),
        (0, 152.034, 7.97051, 4.00628),
    ),
    "SMR_VSA8": (
        scipy.stats.betaprime(0.129794, 2.00845, -1.78743e-34, 0.402201),
        (0, 0, 0, 0),
    ),
    "ExactMolWt": (
        scipy.stats.mielke(6.03051, 6.08107, -3.19057, 393.798),
        (7.01546, 7902.7, 413.218, 196.117),
    ),
    "fr_imidazole": (
        scipy.stats.wald(-0.0177111, 0.0590877),
        (0, 11, 0.102027, 0.359055),
    ),
    "fr_aldehyde": (
        scipy.stats.halflogistic(-2.28021e-10, 0.00326015),
        (0, 2, 3.26023e-3, 0.0575292),
    ),
    "fr_Al_COO": (
        scipy.stats.beta(0.695189, 401.388, -1.84902e-28, 14.9026),
        (0, 9, 0.056974, 0.272083),
    ),
    "NumAliphaticHeterocycles": (
        scipy.stats.alpha(5.57136e-9, -0.0747729, 0.10966),
        (0, 22, 0.754663, 0.918604),
    ),
    "fr_Ar_NH": (
        scipy.stats.wald(-0.0199131, 0.0665108),
        (0, 13, 0.113278, 0.372973),
    ),
    "NumHAcceptors": (
        scipy.stats.logistic(5.03952, 1.27731),
        (0, 199, 5.28545, 3.92937),
    ),
    "fr_lactam": (
        scipy.stats.halflogistic(-1.99948e-10, 1.96e-3),
        (0, 2, 1.96014e-3, 0.0449032),
    ),
    "fr_NH2": (
        scipy.stats.wald(-0.0258865, 0.0866638),
        (0, 17, 0.14403, 0.508089),
    ),
    "fr_Ndealkylation1": (
        scipy.stats.wald(-0.0141703, 0.0471464),
        (0, 8, 0.0843159, 0.35286),
    ),
    "SlogP_VSA7": (
        scipy.stats.recipinvgauss(124626, -5.0391e-11, 0.415794),
        (0, 77.0554, 1.18141, 2.94468),
    ),
    "fr_Ar_N": (
        scipy.stats.halfgennorm(0.323777, -8.49434e-22, 0.0105547),
        (0, 66, 1.35468, 1.77003),
    ),
    "NumSaturatedHeterocycles": (
        scipy.stats.halfgennorm(0.393405, -3.08488e-23, 0.00775584),
        (0, 22, 0.545018, 0.842369),
    ),
    "NumAliphaticRings": (
        scipy.stats.gennorm(0.16215, 1, 1.26442e-6),
        (0, 24, 0.978759, 1.08786),
    ),
    "SMR_VSA4": (
        scipy.stats.betaprime(0.817704, 2.026, -2.70762e-29, 0.895533),
        (0, 169.224, 3.52166, 6.36791),
    ),
    "Chi0v": (
        scipy.stats.mielke(5.87758, 5.96929, -0.0516452, 16.2555),
        (1, 310.367, 17.1295, 7.92387),
    ),
    "qed": (
        scipy.stats.johnsonsb(-0.537684, 0.943839, -0.0597166, 1.027),
        (1.61001e-3, 0.948402, 0.570778, 0.213147),
    ),
    "fr_sulfonamd": (
        scipy.stats.betaprime(0.606187, 2.47005, -1.71097e-30, 0.0241361),
        (0, 2, 0.099297, 0.310673),
    ),
    "fr_halogen": (
        scipy.stats.exponweib(1.59362, 0.477327, -6.30556e-30, 0.111362),
        (0, 22, 0.665657, 1.15388),
    ),
    "Chi4v": (
        scipy.stats.mielke(3.64141, 4.91608, -0.196127, 4.27231),
        (0, 80.3102, 3.9972, 2.18221),
    ),
    "MolLogP": (
        scipy.stats.nct(5.42313, -0.250542, 3.78713, 1.44752),
        (-27.1219, 26.477, 3.35766, 1.85189),
    ),
    "Chi2n": (
        scipy.stats.burr(5.32317, 0.961445, -0.518223, 7.4032),
        (0, 140.422, 7.32038, 3.68307),
    ),
    "fr_Al_OH": (
        scipy.stats.pareto(8.07564, -0.696171, 0.696171),
        (0, 37, 0.189853, 0.580691),
    ),
    "LabuteASA": (
        scipy.stats.mielke(5.90333, 6.09325, -1.16476, 165.594),
        (19.375, 3138.81, 172.786, 78.7224),
    ),
    "SMR_VSA5": (
        scipy.stats.johnsonsu(-6.77038, 1.56398, -8.88574, 0.874722),
        (0, 1059.7, 31.9209, 31.7017),
    ),
    "fr_guanido": (
        scipy.stats.halflogistic(-2.89486e-11, 0.0123907),
        (0, 8, 0.0123909, 0.135269),
    ),
    "SlogP_VSA6": (
        scipy.stats.dweibull(1.26512, 44.8855, 19.9994),
        (0, 425.505, 46.8615, 23.994),
    ),
    "NumRadicalElectrons": (
        scipy.stats.halfnorm(-3.35565e-9, 0.0470122),
        (0, 10, 2.9002e-4, 0.0470114),
    ),
    "HeavyAtomCount": (
        scipy.stats.mielke(5.54294, 6.01292, -0.104757, 28.1915),
        (1, 545, 29.1877, 13.728),
    ),
    "fr_Ar_COO": (
        scipy.stats.pearson3(2.28486, 9.53341e-3, 0.0108913),
        (0, 4, 0.0193214, 0.142787),
    ),
    "fr_ester": (
        scipy.stats.wald(-0.021345, 0.0713267),
        (0, 9, 0.120898, 0.381661),
    ),
    "NumSaturatedCarbocycles": (
        scipy.stats.invweibull(0.68976, -1.68705e-28, 0.0410043),
        (0, 9, 0.166252, 0.504896),
    ),
    "MolMR": (
        scipy.stats.burr(6.17217, 0.893606, -0.626069, 109.362),
        (0, 1943.47, 111.753, 49.2122),
    ),
    "fr_SH": (
        scipy.stats.halflogistic(-5.62731e-10, 2.94017e-3),
        (0, 2, 0.00294021, 0.0577199),
    ),
    "fr_ketone_Topliss": (
        scipy.stats.invweibull(0.784536, -5.10483e-29, 0.013652),
        (0, 5, 0.0521136, 0.247626),
    ),
    "MolWt": (
        scipy.stats.burr(6.2921, 0.944836, -2.57613, 398.415),
        (6.941, 7906.69, 413.675, 196.245),
    ),
    "Kappa1": (
        scipy.stats.fisk(5.09025, 1.5243, 17.4576),
        (1.5974, 452.478, 20.5114, 10.9012),
    ),
    "fr_term_acetylene": (
        scipy.stats.tukeylambda(1.56196, 5.2122e-3, 8.14126e-3),
        (0, 2, 0.0042603, 0.0663488),
    ),
    "Chi0n": (
        scipy.stats.mielke(4.9137, 5.56535, 0.892758, 15.1519),
        (1, 310.367, 16.5657, 7.78969),
    ),
    "SMR_VSA9": (
        scipy.stats.betaprime(0.637152, 0.1653, -1.10963e-26, 8.20083e-3),
        (0, 68.9941, 6.71193, 7.8983),
    ),
    "fr_hdrzine": (
        scipy.stats.genexpon(2.2034, 4.58195e-11, 1.93441, -6.07147e-12, 0.0214182),
        (0, 3, 0.00972068, 0.100231),
    ),
    "PEOE_VSA11": (
        scipy.stats.betaprime(0.534347, 2.03916, -5.64427e-28, 1.3725),
        (0, 130.795, 3.91635, 6.08512),
    ),
    "PEOE_VSA2": (
        scipy.stats.genlogistic(1025.75, -32.0033, 5.42675),
        (0, 436.502, 8.96837, 10.6889),
    ),
    "fr_C_O": (
        scipy.stats.dweibull(0.799241, 1, 1.13764),
        (0, 79, 1.34621, 1.71262),
    ),
    "EState_VSA2": (
        scipy.stats.dgamma(1.62086, 14.7222, 5.9276),
        (0, 348.223, 16.1769, 14.3002),
    ),
    "fr_aryl_methyl": (
        scipy.stats.pareto(2.2591, -0.0348437, 0.0348437),
        (0, 10, 0.42468, 0.733463),
    ),
    "EState_VSA9": (
        scipy.stats.gompertz(1.57763e11, -8.29798e-12, 1.74027e12),
        (0, 459.53, 9.40543, 11.2925),
    ),
    "SlogP_VSA1": (
        scipy.stats.betaprime(24.5554, 8.67678, -8.42638, 5.72863),
        (0, 409.495, 9.98474, 9.77983),
    ),
    "PEOE_VSA9": (
        scipy.stats.betaprime(3.44518, 31.1447, -4.51164, 170.017),
        (0, 811.144, 14.9501, 13.0413),
    ),
    "SMR_VSA2": (
        scipy.stats.wald(-0.0487676, 0.16187),
        (0, 43.2743, 0.299051, 1.36044),
    ),
    "fr_quatN": (
        scipy.stats.halflogistic(-1.83986e-10, 2.28025e-3),
        (0, 2, 0.00228016, 0.050941),
    ),
    "fr_dihydropyridine": (
        scipy.stats.tukeylambda(1.55082, 2.93257e-3, 4.54788e-3),
        (0, 1, 0.00235016, 0.0484215),
    ),
    "MinPartialCharge": (
        scipy.stats.johnsonsu(-2.71495, 1.00384, -0.513906, 8.49505e-3),
        (-0.75391, 1, -0.420078, 0.0718917),
    ),
    "fr_ketone": (
        scipy.stats.wald(-0.01055, 0.0350362),
        (0, 6, 0.0642145, 0.277907),
    ),
    "MaxAbsEStateIndex": (
        scipy.stats.t(1.31085, 12.7535, 0.710667),
        (0, 18.0933, 12.0097, 2.34998),
    ),
    "MaxAbsPartialCharge": (
        scipy.stats.johnsonsu(1.79171, 0.988441, 0.509456, 0.0190802),
        (0.0446722, 3, 0.4247, 0.0715897),
    ),
    "Chi1v": (
        scipy.stats.burr(7.68795, 1.44973, -5.29146, 14.0071),
        (0, 193.522, 10.2333, 4.93729),
    ),
    "fr_benzodiazepine": (
        scipy.stats.tukeylambda(1.56878, 0.00241584, 0.00378991),
        (0, 1, 1.96014e-3, 0.04423),
    ),
    "EState_VSA5": (
        scipy.stats.exponweib(0.267181, 1.16564, -1.38225e-31, 22.7677),
        (0, 352.52, 13.7223, 14.3693),
    ),
    "VSA_EState7": (
        scipy.stats.loggamma(1.65824e-4, 4.69054e-7, 5.38972e-8),
        (-0.239358, 0, -2.39375e-6, 7.5694e-4),
    ),
    "fr_C_O_noCOO": (
        scipy.stats.gennorm(0.306131, 1, 0.00460248),
        (0, 71, 1.26998, 1.62475),
    ),
    "Chi3v": (
        scipy.stats.mielke(4.393, 5.27028, -0.327917, 5.89041),
        (0, 106.268, 5.64645, 2.88191),
    ),
    "PEOE_VSA5": (
        scipy.stats.gibrat(-0.0499382, 0.13664),
        (0, 73.1194, 2.03984, 4.66946),
    ),
    "fr_epoxide": (
        scipy.stats.hypsecant(9.77536e-6, 0.00422932),
        (0, 3, 0.00150011, 0.0412052),
    ),
    "fr_prisulfonamd": (
        scipy.stats.betaprime(0.129794, 2.00845, -1.78743e-34, 0.402201),
        (0, 0, 0, 0),
    ),
    "fr_phenol": (
        scipy.stats.invweibull(0.813599, -4.80616e-28, 0.0162291),
        (0, 8, 0.0502635, 0.253413),
    ),
    "fr_sulfide": (
        scipy.stats.gengamma(1.51063, 0.557426, -4.44139e-30, 0.00383516),
        (0, 6, 0.0810157, 0.288293),
    ),
    "fr_alkyl_halide": (
        scipy.stats.wald(-0.0435805, 0.145232),
        (0, 17, 0.254848, 0.906667),
    ),
    "NumAromaticHeterocycles": (
        scipy.stats.halfgennorm(0.190571, -1.89769e-17, 2.12613e-5),
        (0, 33, 0.945886, 1.02628),
    ),
    "fr_Ar_OH": (
        scipy.stats.wald(-0.0100135, 0.0332426),
        (0, 8, 0.0612443, 0.276613),
    ),
    "fr_thiazole": (
        scipy.stats.wald(-0.00725753, 0.0240681),
        (0, 6, 0.0450232, 0.217339),
    ),
    "fr_imide": (
        scipy.stats.pearson3(2.22216, 0.0180634, 0.0200699),
        (0, 6, 0.0275219, 0.177834),
    ),
    "NumSaturatedRings": (
        scipy.stats.halfgennorm(0.232468, -2.42674e-25, 0.00026458),
        (0, 22, 0.71127, 0.997116),
    ),
    "fr_hdrzone": (
        scipy.stats.wald(-0.00310552, 0.0102744),
        (0, 2, 0.0199514, 0.142805),
    ),
    "fr_lactone": (
        scipy.stats.tukeylambda(1.52715, 0.0135857, 0.0207475),
        (0, 6, 0.0110008, 0.112427),
    ),
    "FractionCSP3": (
        scipy.stats.gausshyper(
            0.477152,
            9.06607,
            -7.62049,
            3.18181,
            -4.69182e-28,
            1.32156,
        ),
        (0, 1, 0.343285, 0.195864),
    ),
    "HallKierAlpha": (
        scipy.stats.logistic(-2.75687, 0.624948),
        (-56.56, 3.03, -2.8237, 1.36225),
    ),
    "fr_para_hydroxylation": (
        scipy.stats.gibrat(-0.00874073, 0.0239887),
        (0, 7, 0.258198, 0.558892),
    ),
    "HeavyAtomMolWt": (
        scipy.stats.burr(6.14824, 0.953966, -2.05973, 373.865),
        (6.941, 7542.8, 389.101, 184.239),
    ),
    "SlogP_VSA12": (
        scipy.stats.lomax(1.35261, -3.84363e-13, 0.530672),
        (0, 199.366, 6.22247, 9.77334),
    ),
    "fr_allylic_oxid": (
        scipy.stats.wald(-0.012126, 0.0402175),
        (0, 12, 0.0752653, 0.433692),
    ),
    "fr_alkyl_carbamate": (
        scipy.stats.wald(-0.00280434, 0.00927635),
        (0, 3, 0.0180713, 0.138365),
    ),
    "fr_HOCCN": (
        scipy.stats.wald(-0.00198261, 0.00655533),
        (0, 2, 0.0128609, 0.113734),
    ),
    "Chi1n": (
        scipy.stats.mielke(4.96026, 5.59423, -0.0629921, 9.41575),
        (0, 179.817, 9.66829, 4.6365),
    ),
    "PEOE_VSA4": (
        scipy.stats.pareto(1.76449, -1.01302, 1.01302),
        (0, 74.6371, 2.87104, 5.25258),
    ),
    "NOCount": (
        scipy.stats.dgamma(0.911489, 6, 2.36084),
        (0, 237, 6.57658, 5.16541),
    ),
    "EState_VSA4": (
        scipy.stats.foldnorm(0.00591151, -9.39602e-10, 29.8685),
        (0, 309.68, 23.7331, 18.1348),
    ),
    "VSA_EState6": (
        scipy.stats.betaprime(0.129794, 2.00845, -1.78743e-34, 0.402201),
        (0, 0, 0, 0),
    ),
    "Chi3n": (
        scipy.stats.mielke(3.8419, 5.02107, -0.128435, 5.26846),
        (0, 89.4191, 5.10249, 2.62333),
    ),
    "fr_barbitur": (
        scipy.stats.genhalflogistic(0.00208257, -0.00141694, 0.00528668),
        (0, 2, 0.0014101, 0.0380541),
    ),
    "fr_Al_OH_noTert": (
        scipy.stats.gompertz(3.51181e7, -1.56833e-10, 5.71218e6),
        (0, 37, 0.169872, 0.555056),
    ),
    "fr_COO2": (
        scipy.stats.wald(-0.0127559, 0.0424195),
        (0, 9, 0.0763453, 0.306787),
    ),
    "fr_azo": (
        scipy.stats.genhalflogistic(4.00027e-4, -2.79684e-5, 0.00310333),
        (0, 2, 6.70047e-4, 0.0270111),
    ),
    "FpDensityMorgan1": (
        scipy.stats.t(7.13574, 1.09714, 0.172135),
        (0.0797101, 2.11111, 1.09477, 0.202011),
    ),
    "fr_aniline": (
        scipy.stats.halfgennorm(0.104917, -0.000421096, 2.05613e-11),
        (0, 17, 0.670157, 0.899993),
    ),
    "SMR_VSA3": (
        scipy.stats.dgamma(1.75255, 12.3495, 4.09735),
        (0, 394.261, 12.6546, 11.2382),
    ),
    "fr_tetrazole": (
        scipy.stats.wald(-0.00180404, 0.00596432),
        (0, 2, 0.0117208, 0.10809),
    ),
    "VSA_EState10": (
        scipy.stats.gennorm(0.325862, -1.58576e-27, 0.00924944),
        (-22.7893, 74.7727, 1.3376, 3.81972),
    ),
    "fr_phenol_noOrthoHbond": (
        scipy.stats.invweibull(0.813603, -5.13795e-29, 0.016108),
        (0, 8, 0.0496235, 0.25148),
    ),
    "PEOE_VSA8": (
        scipy.stats.dgamma(1.4916, 21.4686, 8.38592),
        (0, 311.79, 25.2483, 16.3784),
    ),
    "EState_VSA8": (
        scipy.stats.genexpon(0.849214, 0.898457, 1.87945, -7.99879e-10, 28.404),
        (0, 364.862, 21.0337, 18.3931),
    ),
    "BalabanJ": (
        scipy.stats.nct(4.18266, 2.09653, 1.12718, 0.261649),
        (0, 7.28936, 1.80392, 0.47847),
    ),
    "fr_C_S": (
        scipy.stats.tukeylambda(1.37323, 0.00614215, 0.00843459),
        (0, 2, 0.0173512, 0.133605),
    ),
    "fr_ArN": (
        scipy.stats.tukeylambda(1.44242, 0.0332914, 0.0480201),
        (0, 16, 0.0523937, 0.335936),
    ),
    "NumAromaticRings": (
        scipy.stats.dgamma(2.58517, 2.49545, 0.372622),
        (0, 34, 2.48092, 1.26191),
    ),
    "fr_Imine": (
        scipy.stats.wald(-0.00413358, 0.0136827),
        (0, 4, 0.0263618, 0.17166),
    ),
    "NumAliphaticCarbocycles": (
        scipy.stats.halfgennorm(0.39097, -4.26024e-24, 0.00120957),
        (0, 9, 0.224096, 0.584729),
    ),
    "fr_piperzine": (
        scipy.stats.invweibull(0.788644, -2.42567e-29, 0.0256408),
        (0, 5, 0.0823158, 0.28309),
    ),
    "fr_nitroso": (
        scipy.stats.genhalflogistic(2.1085e-7, -4.92805e-5, 0.00271107),
        (0, 2, 0.00014001, 0.0126488),
    ),
    "FpDensityMorgan2": (
        scipy.stats.johnsonsu(0.711112, 1.75698, 1.98205, 0.356647),
        (0.130435, 2.75, 1.80725, 0.264043),
    ),
    "SlogP_VSA3": (
        scipy.stats.genhalflogistic(0.00150188, -2.31124e-10, 9.57719),
        (0, 486.412, 13.2076, 14.072),
    ),
    "fr_urea": (
        scipy.stats.wald(-0.011156, 0.0370822),
        (0, 4, 0.0670847, 0.258001),
    ),
    "VSA_EState9": (
        scipy.stats.t(2.77851, 54.0047, 14.8416),
        (-61.3539, 1513.33, 58.2982, 32.9722),
    ),
    "fr_nitro_arom": (
        scipy.stats.exponweib(1.13474, 0.764242, -9.54067e-31, 7.66606e-3),
        (0, 4, 0.0315222, 0.185873),
    ),
    "fr_amidine": (
        scipy.stats.gompertz(1.2564e12, -3.63224e-14, 1.72786e10),
        (0, 4, 0.0163211, 0.139768),
    ),
    "fr_nitro_arom_nonortho": (
        scipy.stats.wald(-2.92774e-3, 9.68511e-3),
        (0, 3, 0.0188413, 0.140023),
    ),
    "SlogP_VSA11": (
        scipy.stats.invweibull(0.4147, -1.43253e-27, 0.434835),
        (0, 68.9941, 3.21684, 5.02796),
    ),
    "RingCount": (
        scipy.stats.dgamma(2.18656, 3.48369, 0.497761),
        (0, 57, 3.45968, 1.57758),
    ),
    "fr_azide": (
        scipy.stats.hypsecant(1.11003e-3, 2.81799e-3),
        (0, 2, 9.90069e-4, 0.0323897),
    ),
    "Ipc": (
        scipy.stats.ncf(2.23071, 0.108991, 1, -1.19816e221, 0.559643),
        (0, 9.47613e221, 1.02459e217, math.inf),
    ),
    "fr_benzene": (
        scipy.stats.dgamma(3.31928, 1.50615, 0.239105),
        (0, 14, 1.53443, 0.95617),
    ),
    "fr_thiocyan": (
        scipy.stats.gengamma(0.507695, 1.22545, -1.45433e-31, 8.01926e-3),
        (0, 2, 2.9002e-4, 0.017605),
    ),
    "PEOE_VSA14": (
        scipy.stats.pareto(1.75967, -2.23986, 2.23986),
        (0, 416.498, 3.01375, 6.88319),
    ),
    "PEOE_VSA7": (
        scipy.stats.dgamma(1.48064, 39.9816, 10.1684),
        (0, 508.396, 41.7866, 19.8143),
    ),
    "VSA_EState5": (
        scipy.stats.genhalflogistic(4.73389e-9, -0.030429, 0.0915558),
        (0, 68.1907, 0.00814728, 0.503087),
    ),
    "EState_VSA7": (
        scipy.stats.powerlaw(0.210325, -1.03115e-26, 231.781),
        (0, 225.313, 25.7419, 21.9536),
    ),
    "fr_N_O": (
        scipy.stats.exponnorm(5045.04, -2.41819e-5, 5.7188e-6),
        (0, 6, 0.028882, 0.238642),
    ),
    "VSA_EState4": (
        scipy.stats.betaprime(0.129794, 2.00845, -1.78743e-34, 0.402201),
        (0, 0, 0, 0),
    ),
    "EState_VSA6": (
        scipy.stats.chi(0.538649, -2.64057e-29, 30.8448),
        (0, 298.306, 18.1478, 16.4171),
    ),
    "PEOE_VSA6": (
        scipy.stats.exponpow(0.901621, -5.74834e-27, 53.8242),
        (0, 415.683, 30.514, 23.5803),
    ),
    "fr_diazo": (
        scipy.stats.halfnorm(-8.29974e-10, 0.00316232),
        (0, 1, 1.00007e-5, 0.00316237),
    ),
    "MaxEStateIndex": (
        scipy.stats.t(1.31085, 12.7535, 0.710667),
        (0, 18.0933, 12.0097, 2.34998),
    ),
    "fr_oxime": (
        scipy.stats.pearson3(2.51785, 0.00925187, 0.0116474),
        (0, 3, 0.0154011, 0.126744),
    ),
    "SlogP_VSA10": (
        scipy.stats.betaprime(0.437549, 1.87603, -1.05048e-28, 1.36799),
        (0, 99.4919, 6.34539, 7.37391),
    ),
    "fr_nitrile": (
        scipy.stats.invweibull(0.813386, -2.60923e-30, 0.0105871),
        (0, 5, 0.0447731, 0.221108),
    ),
    "fr_COO": (
        scipy.stats.wald(-0.0127471, 0.0423886),
        (0, 9, 0.0762953, 0.306588),
    ),
    "VSA_EState8": (
        scipy.stats.cauchy(0.461521, 1.4177),
        (-4.31158, 610.905, 9.97152, 18.4882),
    ),
    "SlogP_VSA2": (
        scipy.stats.lognorm(0.511796, -7.27259, 42.8591),
        (0, 1181.09, 41.9594, 34.1554),
    ),
    "fr_priamide": (
        scipy.stats.exponweib(1.09546, 0.754497, -8.87966e-32, 0.0120586),
        (0, 6, 0.0377026, 0.212748),
    ),
    "SMR_VSA1": (
        scipy.stats.cauchy(13.199, 5.44099),
        (0, 611.728, 15.8812, 15.3249),
    ),
    "FpDensityMorgan3": (
        scipy.stats.johnsonsu(1.0544, 1.73487, 2.76302, 0.379821),
        (0.181159, 3.54545, 2.47263, 0.32058),
    ),
    "fr_bicyclic": (
        scipy.stats.beta(0.584006, 474.4, -2.20078e-31, 286.909),
        (0, 31, 0.789255, 1.22226),
    ),
    "TPSA": (
        scipy.stats.johnsonsu(-0.857901, 1.23247, 51.2501, 29.0092),
        (0.0, 3183.79, 83.5727, 74.8475),
    ),
    "NumHeteroatoms": (
        scipy.stats.genlogistic(22.1444, -1.42412, 2.46214),
        (0.0, 259.0, 7.61976, 5.48437),
    ),
    "fr_pyridine": (
        scipy.stats.gibrat(-6.1058e-3, 0.016715),
        (0, 5, 0.220855, 0.481915),
    ),
    "MinEStateIndex": (
        scipy.stats.cauchy(-0.406556, 0.437071),
        (-9.8582, 1.42426, -1.11479, 1.55039),
    ),
    "NumHDonors": (
        scipy.stats.johnsonsu(-6.30379, 1.29357, -0.440635, 0.023159),
        (0, 104, 1.60166, 2.24723),
    ),
    "NumValenceElectrons": (
        scipy.stats.t(2.85425, 144.303, 32.0091),
        (0, 2996, 153.578, 75.1024),
    ),
    "Chi0": (
        scipy.stats.fisk(5.94114, -0.0737428, 19.6546),
        (0, 405.825, 20.8281, 9.92837),
    ),
    "Kappa2": (
        scipy.stats.fisk(4.66843, 0.253901, 7.82235),
        (0.271316, 225.969, 8.91228, 5.46185),
    ),
    "NHOHCount": (
        scipy.stats.invgamma(3.87976, -1.13077, 8.22794),
        (0, 117, 1.74291, 2.56318),
    ),
    "SMR_VSA10": (
        scipy.stats.dweibull(1.15394, 20.8467, 11.836),
        (0, 491.002, 23.9067, 16.4741),
    ),
    "PEOE_VSA12": (
        scipy.stats.alpha(0.542925, -1.18715, 2.14505),
        (0, 419.41, 5.59008, 9.56482),
    ),
    "PEOE_VSA1": (
        scipy.stats.burr(3.9268, 0.489952, -1.64071, 19.2944),
        (0, 561.084, 14.9992, 13.7764),
    ),
    "fr_ether": (
        scipy.stats.fisk(0.805117, -2.56738e-22, 0.237532),
        (0, 60, 0.816737, 1.2247),
    ),
    "EState_VSA1": (
        scipy.stats.foldcauchy(0.00674871, -2.74493e-9, 4.28822),
        (0, 1261.85, 12.5391, 28.9359),
    ),
    "VSA_EState3": (
        scipy.stats.betaprime(0.129794, 2.00845, -1.78743e-34, 0.402201),
        (0, 0, 0, 0),
    ),
    "SlogP_VSA9": (
        scipy.stats.betaprime(0.129794, 2.00845, -1.78743e-34, 0.402201),
        (0, 0, 0, 0),
    ),
    "MaxPartialCharge": (
        scipy.stats.dgamma(0.896643, 0.26139, 0.0674417),
        (-0.0385636, 3, 0.278547, 0.0836452),
    ),
    "BertzCT": (
        scipy.stats.beta(18.189, 6.53839e12, -671.969, 5.92927e14),
        (0, 30826.2, 987.579, 632.153),
    ),
    "fr_isocyan": (
        scipy.stats.genhalflogistic(2.17645e-10, -8.58684e-13, 3.31879e-3),
        (0, 1, 1.10008e-4, 0.0104879),
    ),
    "fr_phos_ester": (
        scipy.stats.genhalflogistic(3.25947e-11, -1.24833e-3, 0.0295266),
        (0, 22, 7.02049e-3, 0.336925),
    ),
    "fr_Nhpyrrole": (
        scipy.stats.wald(-1.99131e-2, 6.65108e-2),
        (0, 13, 0.113278, 0.372973),
    ),
    "RDKit2D_calculated": (
        scipy.stats.betaprime(0.341309, 238.425, 1, 0.883438),
        (True, True, 1, 0),
    ),
    "fr_sulfone": (
        scipy.stats.pearson3(2.1961, 9.51388e-3, 0.0104467),
        (0, 3, 0.0213615, 0.148072),
    ),
    "MinAbsPartialCharge": (
        scipy.stats.laplace(0.263159, 0.0561295),
        (0.000178227, 1, 0.273961, 0.0753377),
    ),
    "SMR_VSA6": (
        scipy.stats.recipinvgauss(0.508265, -6.08695, 8.52444),
        (0, 817.182, 19.2091, 17.4158),
    ),
    "fr_thiophene": (
        scipy.stats.halfgennorm(0.576609, -6.72363e-19, 0.0018219),
        (0, 3, 0.0507436, 0.226691),
    ),
    "EState_VSA11": (
        scipy.stats.genhalflogistic(1.08723e-3, -4.39809e-7, 0.319),
        (0, 163.014, 0.114269, 1.88414),
    ),
    "fr_NH0": (
        scipy.stats.foldnorm(0.0586537, -2.42592e-9, 2.81185),
        (0, 59, 2.17443, 1.78586),
    ),
    "SlogP_VSA5": (
        scipy.stats.genexpon(0.00285259, 0.233537, 0.214434, -2.35732, 4.44721),
        (0, 649.936, 30.7405, 23.895),
    ),
    "EState_VSA10": (
        scipy.stats.genexpon(2.70571, 3.77188, 5.3759, -4.35456e-7, 54.165),
        (0, 322.991, 11.7401, 11.1619),
    ),
    "fr_NH1": (
        scipy.stats.exponnorm(2892.64, -1.29815e-3, 3.79579e-4),
        (0, 70, 1.10729, 1.55904),
    ),
    "SlogP_VSA4": (
        scipy.stats.halfgennorm(0.132903, -2.16371e-12, 4.033e-8),
        (0, 116.324, 6.35858, 7.8001),
    ),
    "fr_Ndealkylation2": (
        scipy.stats.wald(-0.0310692, 0.104438),
        (0, 12, 0.166262, 0.431674),
    ),
    "SMR_VSA7": (
        scipy.stats.dweibull(1.21538, 56.2847, 23.0292),
        (0, 551.786, 58.74, 28.4271),
    ),
    "fr_nitro": (
        scipy.stats.wald(-0.00603098, 0.0199845),
        (0, 4, 0.0378627, 0.20369),
    ),
    "SlogP_VSA8": (
        scipy.stats.ncf(0.0103513, 2.23822, 2.39793, -2.03053e-8, 9.35455e-4),
        (0, 133.902, 6.76662, 8.67185),
    ),
    "VSA_EState2": (
        scipy.stats.genlogistic(1.41561, -4.0305e-5, 1.08857e-3),
        (-0.77269, 0.0802672, -3.274e-5, 4.08318e-3),
    ),
    "NumAromaticCarbocycles": (
        scipy.stats.dgamma(3.31969, 1.50623, 0.239085),
        (0, 14, 1.53504, 0.956148),
    ),
    "fr_furan": (
        scipy.stats.halfgennorm(0.593986, -5.72602e-19, 1.77853e-3),
        (0, 2, 0.0388027, 0.197477),
    ),
    "PEOE_VSA13": (
        scipy.stats.halfgennorm(0.272601, -6.07063e-24, 2.37183e-3),
        (0, 78.1776, 3.3001, 4.35878),
    ),
    "fr_oxazole": (
        scipy.stats.wald(-1.8806e-3, 6.21775e-3),
        (0, 2, 0.0122109, 0.110733),
    ),
    "Kappa3": (
        scipy.stats.foldnorm(0.73819, 0.075, 4.41064),
        (0.075, 1761, 4.87356, 6.73205),
    ),
    "fr_morpholine": (
        scipy.stats.exponweib(1.56837, 0.505355, -2.1401e-30, 3.34665e-3),
        (0, 3, 0.0469233, 0.219777),
    ),
    "fr_unbrch_alkane": (
        scipy.stats.genhalflogistic(0.000173587, -2.40675e-8, 0.255841),
        (0, 167, 0.255588, 1.82168),
    ),
    "fr_amide": (
        scipy.stats.recipinvgauss(220104, -1.4778e-11, 0.745144),
        (0, 71, 1.1423, 1.63981),
    ),
    "NumRotatableBonds": (
        scipy.stats.mielke(2.90296, 4.19919, -0.996823, 8.26713),
        (0, 304, 7.14031, 6.23394),
    ),
    "Chi1": (
        scipy.stats.mielke(5.37253, 6.06837, -0.0679089, 13.7257),
        (0, 256.97, 14.0335, 6.54332),
    ),
    "fr_phos_acid": (
        scipy.stats.genhalflogistic(3.25947e-11, -0.00124833, 0.0295266),
        (0, 22, 7.1305e-3, 0.337501),
    ),
    "fr_piperdine": (
        scipy.stats.halfgennorm(0.423879, -1.27965e-22, 0.00187708),
        (0, 6, 0.14572, 0.391648),
    ),
    "fr_isothiocyan": (
        scipy.stats.genhalflogistic(8.88666e-9, -0.000892495, 0.00268529),
        (0, 2, 2.30016e-4, 0.0158103),
    ),
    "MinAbsEStateIndex": (
        scipy.stats.genpareto(0.116543, -4.29541e-11, 0.132457),
        (0, 2.51157, 0.150168, 0.172921),
    ),
    "fr_methoxy": (
        scipy.stats.gibrat(-0.0155091, 0.0428669),
        (0, 8, 0.313382, 0.652371),
    ),
}
"""A mapping from a RDKit descriptor name to a 2-tuple of its parameterized distribution and a tuple of its clipping
parameters."""

DESCRIPTOR_DISTRIBUTIONS: Mapping[str, tuple[SupportsCDF, tuple[float, ...]]] = MappingProxyType(
    DESCRIPTOR_DISTRIBUTIONS
)
