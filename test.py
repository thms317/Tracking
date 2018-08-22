import numpy as np

matrix = [2.4849633423955332e-05, 1.2130340175935334e-05, 7.2708599034922844e-05, 1.2255545200538677e-05, 0.27866974395770633, 1.9890336029025403e-06, 7.5389073344719451e-05, 5.6190203740450688e-05, 0.26834723628781498, 1.2793465528914535e-05, 6.6520688395853593e-05, 0.00021740303470009891, 1.1503545874480303, 0.84862342429445725, 2.8391015816810697e-05, 4.8469896117606538e-05, 3.9600293657426636e-05, 1.7575493133101858e-05, 3.6700623267215302e-05, 2.645145414094384e-05, 2.0411574266386039e-05, 1.5090061754222713e-05, 2.648459435257531e-05, 3.464790164126137e-05, 0.62473054895609303, 3.2961681870547356e-05, 2.3165721079549758e-05, 3.2616091820468459e-05, 3.4229225940585625e-05, 2.8000555886779827e-05, 3.1713488625138885e-05, 2.7312631888030997e-05, 2.6856587753141433e-05, 1.4236294714132078e-05, 2.2121034554412876e-05, 2.3081480967226375e-05, 2.440004628224082e-05, 2.2720794158047384e-05, 0.2292438590579467, 3.1654384093664804e-05, 2.8271335826939168e-05, 1.7086259544148921e-05, 2.8010759071457269e-05, 1.8544730140282233e-05, 2.4839964584812124e-05, 1.7341118386473719e-05, 2.0978564961652932e-05, 8.8614368009002004e-06, 2.4384988786046604e-05, 1.8526632276873878e-05, 1.8792092093918153e-05, 1.5903293379396887e-05, 8.757331798194845e-06, 3.3478528176490649e-05, 2.1543855470896009e-05, 1.5880153562773368e-05, 0.2771783960366988, 0.22132303311748244, 0.21512450224069091, 1.0866659200598696, 0.25893776483526576, 0.036841985851595808, 0.0431464253460703, 0.038326544177971551, 0.23997396728804135, 0.16954930830866607, 0.32897948589280684, 0.2537322366027947, 0.050257541736059903, 0.036628328301993279, 0.24322407484787348, 6.0734428251150787e-05, 6.1618626566068593e-05, 0.31816792955405321, 5.1887377234318618e-05, 4.7904304072476678e-05, 5.9038515060458652e-05, 0.43366016466875196, 4.872844382470069e-05, 5.5502467545339573e-05, 0.30968743451357345, 3.9161270824981886e-05, 0.350097285455228, 6.7075374652806111e-05, 4.9241648672242614e-05, 0.47748296961552394, 3.5003875274202282e-05, 2.9477561597699977e-05, 3.3723685934576414e-05, 0.51734467826538799, 2.8268741984819365e-05, 2.4500888910419989e-05, 2.5956866774828771e-05, 0.55197937567341226, 1.6404317882171738e-05, 5.3400741590768629e-05, 2.9103143876008234e-05, 0.52855294742059789, 0.30991163767195973, 2.8918578503357682e-05, 0.014031719764365692, 3.9101306957492184e-05, 2.7475102327217667e-05, 1.3749452317729316e-05, 0.24363457075177342, 0.25351851547074877, 2.5810778250918796e-05, 2.516861893994778e-05, 1.6667304738471953e-05, 0.40080266024066946, 0.30993545587738658, 2.4381988492845607e-05, 1.2944570136276889, 2.5306484044950665e-05, 0.27096085696088923, 1.4106041290889511, 3.072331740384882e-05, 9.227346834397831e-06, 6.8691427052400031e-05, 0.41002861409178992, 3.0709955453280431e-05, 2.3766358482958937e-05, 1.7506360304355292e-05, 0.00018920314224724389, 2.6778542971043597, 1.6440842674138834e-05]
n = 8
beads = 14
A = ['070', '070', '070', '075', '075', '075', '080', '080', '080']
B = ['070', '075', '080', '070', '075', '080', '070', '075', '080']
freqsA = len(np.unique(A))
freqsB = len(np.unique(B))

matrix = np.array(matrix)
matrix_2D_n_beads = matrix.reshape(n+1,beads)
matrix_3D_freqsA_freqsB_beads = matrix.reshape(freqsA,freqsB,beads)
matrix_3D_beads = np.transpose(matrix.reshape(beads,n+1))
# print(matrix_3D_beads)


# ref_A = 8
# ref_B = 7.5
#
# print(matrix_3D_freqsA_freqsB_beads[(ref_A-7)/0.5][(ref_B-7)/0.5])

N = 4000

for i in range(int(N/2)):
    m = 2**i
    # print(m)
    if m > N/2:
        break

a = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
b = [[2.2720649423966805e-07, 2.4179243380467244e-07, 2.5650650355234932e-07, 3.1012912582819773e-07, 4.3965269065656993e-07, 6.4201152119958121e-07, 1.0959436825753319e-06, 2.7417666638596148e-06, 9.461348254329718e-06, 3.3568305190770606e-05, 0.00013180780900840628, 0.00050193336905482433], [2.1118276509893918e-07, 2.3979201349135534e-07, 2.533248191602831e-07, 3.0294083824816658e-07, 4.2055242157790997e-07, 5.9891833594897274e-07, 9.2502150378600264e-07, 1.714036378742161e-06, 4.9534066755428664e-06, 1.4620602064300793e-05, 3.1948980352455255e-05, 0.00011939028762349072], [3.3360089807445314e-07, 3.7235170806133146e-07, 4.0963917799169573e-07, 4.8878390207573346e-07, 7.4556954145414175e-07, 1.3074468682667607e-06, 2.7720930619415527e-06, 7.7000632296070148e-06, 2.7252677225682975e-05, 9.0330943136317238e-05, 0.00024129349277223378, 0.00076573631930721374], [1.9057143444488246e-07, 2.0201626627399884e-07, 2.1020583979067489e-07, 2.4149494655708644e-07, 3.3948060066081601e-07, 5.2548326617395257e-07, 8.3427858056606419e-07, 1.6927981835491755e-06, 5.0840512525871022e-06, 1.3785313016072481e-05, 2.2260528596775723e-05, 1.7950046402723332e-06], [0.00071704710963600086, 0.0011159687857228066, 0.0013799173954212071, 0.0029544219861049551, 0.0069833190217910394, 0.015040849013227166, 0.031284857965853806, 0.065195526920014235, 0.14040438273640865, 0.32817011329069362, 0.97117569790887981, 3.0586609749103615], [1.2719476988174169e-07, 1.4772759483033999e-07, 1.5881217194507375e-07, 1.9028423661104107e-07, 2.9338215048578871e-07, 4.1475117952695139e-07, 6.1933408839897002e-07, 1.0882976962388776e-06, 1.8142886218498915e-06, 2.5568338497934993e-06, 1.2571589292280259e-06, 9.2293802287683745e-07], [3.2824297491889387e-07, 3.8314105374292194e-07, 4.4187878473238828e-07, 5.5412509738667287e-07, 9.0014454990397526e-07, 1.639516644252802e-06, 3.3436065384288907e-06, 8.332958194341064e-06, 2.8154298605616587e-05, 9.9144398682628846e-05, 0.00031765855594574009, 0.0012917454279283919], [1.0231150149516424e-07, 1.0828353214948115e-07, 1.075802292435556e-07, 1.3450255405091779e-07, 2.3002573638816682e-07, 5.2160568436681374e-07, 1.4993514652734311e-06, 5.2481971011421689e-06, 2.0328017039655009e-05, 7.9560618803353804e-05, 0.00031628447858201664, 0.0011308588154573457], [0.001444636220039844, 0.0019280887644883831, 0.001738050861661698, 0.0017765664069813031, 0.005114131175604642, 0.013639734924679993, 0.030602965871769257, 0.06514883734651887, 0.14104599858031949, 0.33006191502893134, 0.97696153016491738, 3.0594131875161024], [8.1596832048905418e-08, 8.124144457632245e-08, 8.7865983499867301e-08, 1.334088635525809e-07, 2.4536681545905868e-07, 4.3261703641856981e-07, 7.7988357473121513e-07, 1.664728334806855e-06, 5.0042837564848065e-06, 1.6014055413062427e-05, 4.1709644666511922e-05, 5.9858545236674916e-05], [7.6402299745474515e-08, 8.4784593954598899e-08, 9.1598772204816053e-08, 1.3392138783975325e-07, 2.5902563005961067e-07, 6.080640394957714e-07, 1.6809520928607923e-06, 6.0870567504864558e-06, 2.3905136035501127e-05, 9.4507859037822533e-05, 0.00038035668822345033, 0.0014630247019572537], [0.0015548665085970404, 0.0017664358850879404, 0.00074777458401561428, 0.00036020254029040584, 0.00045460814700938644, 0.00015948559567494281, 4.422249054083721e-05, 1.805311347248888e-05, 2.4308348701704086e-05, 7.0385180028497302e-05, 0.00021411383831998902, 0.00089555700689441739], [0.02629792671841967, 0.022730029278241817, 0.01975940562206294, 0.021508686430643749, 0.037701344852140606, 0.079519028924823404, 0.16407409633155445, 0.24883699660328848, 0.5795662517814546, 1.5443632924755952, 1.3015775637501334, 0.0084990226544606126], [0.027740685094448159, 0.018024081745456357, 0.014859583178334416, 0.010770249496638037, 0.0083621722215492834, 0.016699070765261299, 0.051126020393967804, 0.15328706678921403, 0.38045304012507397, 0.90018992853640289, 0.83371306911528931, 1.0540534821591985]]
b = np.array(b)
c = np.vstack([a,b])
print(c)

#
# c = c.reshape(len(a),len(b))
# print(c)