
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Generate universe variables
#   * Quality and service on subjective ranges [0, 10]
#   * Tip has a range of [0, 25] in units of percentage points
x_luas = np.arange(0, 61, 1)
x_curah_hujan= np.arange(0,1,0.01)
x_siaga= np.arange(0,201,1)

#Fuzzy Membership
#Input
luas_sempit = fuzz.trimf(x_luas, [0, 0, 40])
luas_sedang = fuzz.trimf(x_luas, [25,40,55])
luas_lebar = fuzz.trimf(x_luas,[50,60,60])

#Input
curah_gerimis = fuzz.trimf(x_curah_hujan, [0, 0, 0.05])
curah_sedang = fuzz.trimf(x_curah_hujan, [0, 0.025, 0.25])
curah_deras = fuzz.trimf(x_curah_hujan, [0.1, 0.595,1])
curah_badai = fuzz.trimf(x_curah_hujan, [0.595,1,1])

#Output
siaga4 = fuzz.trimf(x_siaga, [40, 80, 115])
siaga3 = fuzz.trimf(x_siaga, [80, 115, 175])
siaga2 = fuzz.trimf(x_siaga, [150, 175, 200])
siaga1 = fuzz.trimf(x_siaga, [175 ,200, 200])

# Graf Membership Function

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(x_luas,luas_sempit, 'b' , linewidth=1.5,label = "sempit")
ax0.plot(x_luas,luas_sedang, 'g' , linewidth=1.5,label = "sedang")
ax0.plot(x_luas,luas_lebar, 'r' , linewidth=1.5,label = "lebar")
ax0.set_title('luas penampang (m^2)')
ax0.legend()

ax2.plot(x_curah_hujan, curah_gerimis, 'b', linewidth=1.5, label='Gerimis')
ax2.plot(x_curah_hujan, curah_sedang ,'g', linewidth=1.5, label='Sedang')
ax2.plot(x_curah_hujan, curah_deras, 'r', linewidth=1.5, label='Deras')
ax2.plot(x_curah_hujan, curah_badai, 'y', linewidth=1.5, label='Badai')
ax2.set_title('Curah Hujan (mm/menit)')
ax2.legend()

ax1.plot(x_siaga, siaga4, 'b', linewidth=1.5, label='Siaga4')
ax1.plot(x_siaga, siaga3, 'g', linewidth=1.5, label='Siaag3')
ax1.plot(x_siaga, siaga2, 'r', linewidth=1.5, label='Siaga2')
ax1.plot(x_siaga, siaga1, 'y', linewidth=1.5, label='Siaga1')
ax1.set_title('Siaga (Berdasarkan Tinggi Air dari Batas Normal) (cm)')
ax1.legend()



# Hapus Border
for ax in (ax0, ax2, ax1):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()


#Fuzzy Rules

inputluas = float(input('Luas Penampang (m^2) = '))
inputcurah = float(input('Curah Hujan (mm/menit)= '))

luas_level_sempit = fuzz.interp_membership(x_luas, luas_sempit, inputluas )
luas_level_sedang = fuzz.interp_membership(x_luas, luas_sedang, inputluas )
luas_level_lebar = fuzz.interp_membership(x_luas, luas_lebar, inputluas)



curah_level_gerimis = fuzz.interp_membership(x_curah_hujan, curah_gerimis,inputcurah)
curah_level_sedang = fuzz.interp_membership(x_curah_hujan, curah_sedang,inputcurah)
curah_level_deras = fuzz.interp_membership(x_curah_hujan, curah_deras,inputcurah)
curah_level_badai = fuzz.interp_membership(x_curah_hujan, curah_badai,inputcurah)

"""
Rule Siaga 1 =
IF ( (LP = SEMPIT & (CH = BADAI OR DERAS)) OR (LP = SEDANG & CH = BADAI )) then Siaga 1
"""
#Rule 1
ch_combine_1 = np.fmax(curah_level_badai, curah_level_deras)
sempit_combine_1 = np.fmin(ch_combine_1,luas_level_sempit)
sedang_badai_1 = np.fmin(luas_level_sedang, curah_level_badai)
active_rule_1 = np.fmax(sedang_badai_1, sempit_combine_1)
siaga1_activation = np.fmin(active_rule_1, siaga1)


"""Rule Siaga 2 =
IF (LP = SEMPIT & CH = SEDANG) OR (LP = SEDANG & CH = DERAS ) OR (LP = LEBAR & CH=BADAI) then Siaga2
"""

sempit_combine_2 = np.fmin(luas_level_sempit, curah_level_sedang)
sedang_combine_2 = np.fmin(luas_level_sedang, curah_level_deras)
lebar_combine_2 = np.fmin(luas_level_lebar, curah_level_badai)
OR_awal_2 = np.fmax(sempit_combine_2,sedang_combine_2)
active_rule_2 = np.fmax(OR_awal_2, lebar_combine_2)
siaga2_activation = np.fmin(active_rule_2, siaga2)


"""Rule Siaga 3 =
IF (LP = SEMPIT & CH = GERIMIS) OR (LP = SEDANG & CH = SEDANG ) OR (LP = LEBAR & CH=DERAS) then Siaga3
"""
sempit_combine_3 = np.fmin(luas_level_sempit, curah_level_gerimis)
sedang_combine_3 = np.fmin(luas_level_sedang, curah_level_sedang)
lebar_combine_3 = np.fmin(luas_level_lebar, curah_level_deras)
OR_awal_3 = np.fmax(sempit_combine_3,sedang_combine_3)
active_rule_3 = np.fmax(OR_awal_3, lebar_combine_3)
siaga3_activation = np.fmin(active_rule_3, siaga3)

"""Rule Siaga 4 =
IF (LP = SEDANG & CH = GERIMIS ) OR ( LP = LEBAR & ( CH = GERIMIS OR SEDANG )) then Siaga4
"""
ch_combine_4 = np.fmax(curah_level_gerimis,curah_level_sedang)
lebar_combine_4 = np.fmin(luas_level_lebar, ch_combine_4)
sedang_combine_4 = np.fmin(luas_level_sedang,curah_level_gerimis)
active_rule_4 = np.fmax(sedang_combine_4,lebar_combine_4)
siaga4_activation = np.fmin(active_rule_4,siaga4)


siaga0 = np.zeros_like(x_siaga)



# Tampilkan Graf Warna Warni
fig, ax0 = plt.subplots(figsize=(10, 3))

ax0.fill_between(x_siaga, siaga0, siaga4_activation, facecolor='b', alpha=0.7)
ax0.plot(x_siaga, siaga4, 'b', linewidth=0.5, label = 'siaga 4', linestyle='--' )
ax0.fill_between(x_siaga, siaga0, siaga3_activation, facecolor='g', alpha=0.7)
ax0.plot(x_siaga, siaga3, 'g', linewidth=0.5, label = 'siaga 3', linestyle='--')
ax0.fill_between(x_siaga, siaga0, siaga2_activation, facecolor='r', alpha=0.7)
ax0.plot(x_siaga, siaga2, 'r', linewidth=0.5, label = 'siaga 2', linestyle='--')
ax0.fill_between(x_siaga, siaga0, siaga1_activation, facecolor='y', alpha=0.7)
ax0.plot(x_siaga, siaga1, 'y', linewidth=0.5,label = 'siaga 1', linestyle='--')
ax0.set_title('Output membership activity')
ax0.legend()

# Hapus Border
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()


# Diagregasikan
aggregated = np.fmax(siaga1_activation,np.fmax(siaga2_activation,np.fmax(siaga3_activation, siaga4_activation)))
# Hasil Defuzifikasi
siaga = fuzz.defuzz(x_siaga, aggregated, 'centroid')
siaga_activation = fuzz.interp_membership(x_siaga, aggregated, siaga)  # for plot

# Tampilkan ( Diagram Oranye )
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(x_siaga, siaga4, 'b', linewidth=0.5, label = 'siaga4',linestyle='--', )
ax0.plot(x_siaga, siaga3, 'g', linewidth=0.5, label = 'siaga3',linestyle='--')
ax0.plot(x_siaga, siaga2, 'r', linewidth=0.5, label = 'siaga2',linestyle='--')
ax0.plot(x_siaga, siaga1, 'y', linewidth=0.5, label = 'siaga1',linestyle='--')
ax0.fill_between(x_siaga, siaga0, aggregated, facecolor='Orange', alpha=0.7)
ax0.plot([siaga, siaga], [0, siaga_activation], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('Hasil Agregasi . Siaga (Berdasarkan Tinggi Air dari Batas Normal) (cm)')
ax0.legend()

# Hapus Border

for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

