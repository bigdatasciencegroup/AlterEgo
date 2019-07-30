import numpy as np
import os.path

from processing import process, process_scrambled, process_silence_between, process_scrambled_flattened, process_file, split, get_inputs
import transform
import serial_data as serial
import from_file

module = os.path.abspath(os.path.dirname(__file__))

def flatten(sequence_groups):
    return np.concatenate(sequence_groups, axis=0)

def combine(datasets):
    assert len(set(map(len, datasets))) == 1
    return [reduce(lambda a,b: list(a)+list(b),
                   map(lambda sg: sg[i], datasets)) for i in range(len(datasets[0]))]

def join(datasets):
    return reduce(lambda a,b: list(a)+list(b), datasets)

def digits_session_0_dataset(**kwargs):
    return join([
            process(1, [os.path.join(module, 'data/8_subvocal_0_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/9_subvocal_1_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/10_subvocal_2_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/13_subvocal_3_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/12_subvocal_4_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/14_subvocal_5_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/15_subvocal_6_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/16_subvocal_7_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/17_subvocal_8_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/19_subvocal_9_50_trials.txt')], **kwargs),
        ])

def digits_session_1_dataset(**kwargs):
    return join([
            process(1, [os.path.join(module, 'data/46_subvocal_0_37_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/55_subvocal_1_35_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/47_subvocal_2_38_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/48_subvocal_3_38_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/49_subvocal_4_40_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/50_subvocal_5_37_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/51_subvocal_6_40_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/52_subvocal_7_44_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/53_subvocal_8_39_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/54_subvocal_9_42_trials.txt')], **kwargs),
        ])

def digits_session_2_dataset(**kwargs):
    return join([
            process(1, [os.path.join(module, 'data/59_subvocal_0_158_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/58_subvocal_1_116_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/60_subvocal_2_186_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/61_subvocal_3_193_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/62_subvocal_4_175_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/63_subvocal_5_203_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/64_subvocal_6_166_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/65_subvocal_7_176_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/66_subvocal_8_181_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/67_subvocal_9_165_trials.txt')], **kwargs),
        ])

def digits_session_3_dataset(**kwargs):
    return combine([
            process(10, [os.path.join(module, 'data/70_subvocal_digits_15_trials.txt')], **kwargs),
            process(10, [os.path.join(module, 'data/71_subvocal_digits_15_trials.txt')], **kwargs),
            process(10, [os.path.join(module, 'data/72_subvocal_digits_14_trials.txt')], **kwargs),
            process(10, [os.path.join(module, 'data/73_subvocal_digits_6_trials.txt')], **kwargs),
    ])

def digits_session_4_dataset(**kwargs):
    return join([
            process(1, [os.path.join(module, 'data/75_subvocal_0_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/76_subvocal_1_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/77_subvocal_2_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/78_subvocal_3_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/79_subvocal_4_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/80_subvocal_5_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/81_subvocal_6_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/82_subvocal_7_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/83_subvocal_8_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/84_subvocal_9_50_trials.txt')], **kwargs),
    ])

def digits_session_5_dataset(**kwargs):
    return join([
            process(1, [os.path.join(module, 'data/85_subvocal_0_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/86_subvocal_1_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/87_subvocal_2_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/88_subvocal_3_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/89_subvocal_4_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/90_subvocal_5_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/91_subvocal_6_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/92_subvocal_7_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/93_subvocal_8_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/94_subvocal_9_50_trials.txt')], **kwargs),
    ])

def digits_session_6_dataset(**kwargs):
    return join([
            process(1, [os.path.join(module, 'data/97_subvocal_0_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/98_subvocal_1_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/99_subvocal_2_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/100_subvocal_3_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/101_subvocal_4_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/102_subvocal_5_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/103_subvocal_6_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/104_subvocal_7_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/105_subvocal_8_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/106_subvocal_9_50_trials.txt')], **kwargs),
    ])

def digits_session_7_dataset(**kwargs):
    return join([
            process(1, [os.path.join(module, 'data/109_subvocal_0_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/110_subvocal_1_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/111_subvocal_2_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/112_subvocal_3_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/113_subvocal_4_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/114_subvocal_5_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/115_subvocal_6_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/116_subvocal_7_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/117_subvocal_8_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/118_subvocal_9_50_trials.txt')], **kwargs),
    ])

def digits_session_8_dataset(**kwargs):
    return join([
            process(1, [os.path.join(module, 'data/120_subvocal_0_30_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/121_subvocal_1_30_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/122_subvocal_3_30_trials.txt')], **kwargs),
    ])

def digits_session_9_dataset(**kwargs):
    return join([
            process(1, [os.path.join(module, 'data/124_subvocal_0_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/125_subvocal_1_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/126_subvocal_9_50_trials.txt')], **kwargs),
    ])

def digits_session_10_dataset(**kwargs):
    return join([
            process(1, [os.path.join(module, 'data/129_subvocal_0_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/130_subvocal_1_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/131_subvocal_9_50_trials.txt')], **kwargs),
    ])

def digits_session_11_dataset(**kwargs):
    return join([
            process(1, [os.path.join(module, 'data/133_subvocal_0_10_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/134_subvocal_1_10_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/135_subvocal_9_10_trials.txt')], **kwargs),
    ])

def digits_session_12_dataset(**kwargs):
    return join([
            process(1, [os.path.join(module, 'data/136_subvocal_one_30_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/137_subvocal_sil_30_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/138_subvocal_medialab_30_trials.txt')], **kwargs),
    ])

def digits_session_13_dataset(**kwargs):
    return join([
            process(1, [os.path.join(module, 'data/140_subvocal_one_50_trials.txt')], **kwargs),
#            process(1, [os.path.join(module, 'data/141_subvocal_sil_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/137_subvocal_sil_30_trials.txt')], **kwargs),
#            process(1, [os.path.join(module, 'data/142_subvocal_media_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/144_subvocal_lala_50_trials.txt')], **kwargs),
    ])

def digits_session_14_dataset(**kwargs):
    return join([
            process(1, [os.path.join(module, 'data/147_subvocal_one_50_trials.txt')], **kwargs),
#            process(1, [os.path.join(module, 'data/148_subvocal_lala_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/151_subvocal_see_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/149_subvocal_medialab_50_trials.txt')], **kwargs),
    ])

def digits_session_15_dataset(**kwargs):
    return join([
            combine([
                    process(1, [os.path.join(module, 'data/153_subvocal_one_50_trials.txt')], **kwargs),
                    process(1, [os.path.join(module, 'data/157_subvocal_one_20_trials.txt')], **kwargs),
                    process(1, [os.path.join(module, 'data/160_subvocal_one_50_trials.txt')], **kwargs),
                ]),
            combine([
                    process(1, [os.path.join(module, 'data/154_subvocal_see_50_trials.txt')], **kwargs),
                    process(1, [os.path.join(module, 'data/158_subvocal_see_20_trials.txt')], **kwargs),
                ]),
            process(1, [os.path.join(module, 'data/155_subvocal_lab_50_trials.txt')], **kwargs),
#            process(1, [os.path.join(module, 'data/159_subvocal_medialab_50_trials.txt')], **kwargs),
    ])

def digits_session_dependence_1_dataset(**kwargs):
    return join([
            process(1, [os.path.join(module, 'data/1_0.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/1_1.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/1_2.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/1_3.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/1_4.txt')], sample_rate=1000, **kwargs),
    ])

def digits_session_dependence_2_dataset(**kwargs):
    return join([
            process(1, [os.path.join(module, 'data/2_0.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/2_1.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/2_2.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/2_3.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/2_4.txt')], sample_rate=1000, **kwargs),
    ])

def digits_session_dependence_3_dataset(**kwargs):
    return join([
            process(1, [os.path.join(module, 'data/3_0.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/3_1.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/3_2.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/3_3.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/3_4.txt')], sample_rate=1000, **kwargs),
    ])

def digits_sequences_session_1_dataset(**kwargs):
    return join([
            process(1, [os.path.join(module, 'data/161_012_30_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/162_120_30_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/163_201_30_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/164_021_30_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/165_210_30_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/166_102_30_trials.txt')], **kwargs),
    ])

def digits_sequences_session_2_dataset(**kwargs):
    return join([
            process(1, [os.path.join(module, 'data/168_012_1k_30.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/169_120_1k_30.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/170_201_1k_30.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/171_021_1k_30.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/172_210_1k_30.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/173_102_1k_30.txt')], sample_rate=1000, **kwargs),
    ])

def phonemes_5_sentences_dataset(**kwargs):
    return join([
            process(1, [os.path.join(module, 'data/175_phoneme5_sen1_30.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/176_phoneme5_sen2_30.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/177_phoneme5_sen3_30.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/178_phoneme5_sen4_30.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/179_phoneme5_sen5_30.txt')], sample_rate=1000, **kwargs),
    ])

def phonemes_30_sentences_dataset(**kwargs):
    return join([
            process(1, [os.path.join(module, 'data/184_p30_s1.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/185_p30_s2.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/186_p30_s3.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/187_p30_s4.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/188_p30_s5.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/189_p30_s6.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/190_p30_s7.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/191_p30_s8.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/192_p30_s9.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/193_p30_s10.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/194_p30_s11.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/195_p30_s12.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/196_p30_s13.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/197_p30_s14.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/198_p30_s15.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/199_p30_s16.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/200_p30_s17.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/201_p30_s18.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/202_p30_s19.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/203_p30_s20.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/204_p30_s21.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/205_p30_s22.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/206_p30_s23.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/207_p30_s24.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/208_p30_s25.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/209_p30_s26.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/210_p30_s27.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/211_p30_s28.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/212_p30_s29.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/213_p30_s30.txt')], sample_rate=1000, **kwargs),
    ])

def words_10_20_sentences_dataset(**kwargs):
    return join([
            process(1, [os.path.join(module, 'data/214_w10_s1.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/215_w10_s2.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/216_w10_s3.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/217_w10_s4.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/218_w10_s5.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/219_w10_s6.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/220_w10_s7.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/221_w10_s8.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/222_w10_s9.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/223_w10_s10.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/224_w10_s11.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/225_w10_s12.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/226_w10_s13.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/227_w10_s14.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/228_w10_s15.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/229_w10_s16.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/230_w10_s17.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/231_w10_s18.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/232_w10_s19.txt')], sample_rate=1000, **kwargs),
            process(1, [os.path.join(module, 'data/233_w10_s20.txt')], sample_rate=1000, **kwargs),
    ])

def words_20_25_sentences_dataset(**kwargs):
    return join([
            process(1, [os.path.join(module, 'data/234_w20_s1.txt')], sample_rate=250, **kwargs),
            process(1, [os.path.join(module, 'data/235_w20_s2.txt')], sample_rate=250, **kwargs),
            process(1, [os.path.join(module, 'data/236_w20_s3.txt')], sample_rate=250, **kwargs),
            process(1, [os.path.join(module, 'data/237_w20_s4.txt')], sample_rate=250, **kwargs),
            process(1, [os.path.join(module, 'data/238_w20_s5.txt')], sample_rate=250, **kwargs),
            process(1, [os.path.join(module, 'data/239_w20_s6.txt')], sample_rate=250, **kwargs),
            process(1, [os.path.join(module, 'data/240_w20_s7.txt')], sample_rate=250, **kwargs),
            process(1, [os.path.join(module, 'data/241_w20_s8.txt')], sample_rate=250, **kwargs),
            process(1, [os.path.join(module, 'data/242_w20_s9.txt')], sample_rate=250, **kwargs),
            process(1, [os.path.join(module, 'data/243_w20_s10.txt')], sample_rate=250, **kwargs),
            process(1, [os.path.join(module, 'data/244_w20_s11.txt')], sample_rate=250, **kwargs),
            process(1, [os.path.join(module, 'data/245_w20_s12.txt')], sample_rate=250, **kwargs),
            process(1, [os.path.join(module, 'data/246_w20_s13.txt')], sample_rate=250, **kwargs),
            process(1, [os.path.join(module, 'data/247_w20_s14.txt')], sample_rate=250, **kwargs),
            process(1, [os.path.join(module, 'data/248_w20_s15.txt')], sample_rate=250, **kwargs),
            process(1, [os.path.join(module, 'data/249_w20_s16.txt')], sample_rate=250, **kwargs),
            process(1, [os.path.join(module, 'data/250_w20_s17.txt')], sample_rate=250, **kwargs),
            process(1, [os.path.join(module, 'data/251_w20_s18.txt')], sample_rate=250, **kwargs),
            process(1, [os.path.join(module, 'data/252_w20_s19.txt')], sample_rate=250, **kwargs),
            process(1, [os.path.join(module, 'data/253_w20_s20.txt')], sample_rate=250, **kwargs),
            process(1, [os.path.join(module, 'data/254_w20_s21.txt')], sample_rate=250, **kwargs),
            process(1, [os.path.join(module, 'data/255_w20_s22.txt')], sample_rate=250, **kwargs),
            process(1, [os.path.join(module, 'data/256_w20_s23.txt')], sample_rate=250, **kwargs),
            process(1, [os.path.join(module, 'data/257_w20_s24.txt')], sample_rate=250, **kwargs),
            process(1, [os.path.join(module, 'data/258_w20_s25.txt')], sample_rate=250, **kwargs),
    ])

def digits_5_1_dataset(**kwargs):
    labels = [1, 4, 4, 3, 1, 4, 1, 4, 3, 0, 4, 3, 2, 3, 2, 1, 3, 1, 0, 1, 4, 0, 1, 3, 2, 2, 4, 4, 0, 4, 4, 0, 1, 4, 3, 4, 0, 0, 0, 4, 4, 0, 1, 2, 0, 2, 0, 0, 0, 0, 4, 1, 2, 2, 3, 2, 2, 0, 0, 2, 3, 3, 3, 3, 0, 3, 0, 3, 2, 1, 1, 4, 3, 3, 3, 1, 1, 4, 3, 4, 1, 0, 3, 0, 1, 4, 1, 2, 1, 1, 2, 3, 1, 2, 0, 0, 0, 3, 2, 0, 4, 1, 1, 4, 2, 4, 1, 4, 1, 4, 2, 2, 2, 1, 1, 3, 2, 4, 1, 3, 0, 3, 4, 3, 4, 3, 4, 3, 1, 2, 3, 3, 0, 4, 3, 0, 2, 2, 0, 4, 4, 3, 4, 1, 3, 4, 4, 0, 3, 0, 4, 3, 1, 2, 1, 3, 0, 1, 1, 3, 1, 2, 3, 0, 3, 0, 2, 3, 1, 3, 3, 3, 2, 2, 0, 2, 0, 4, 2, 3, 3, 2, 2, 4, 1, 0, 0, 4, 2, 1, 1, 0, 1, 0, 0, 0, 4, 0, 2, 3, 2, 4, 0, 4, 2, 3, 2, 4, 0, 4, 1, 2, 0, 1, 0, 1, 1, 0, 4, 2, 1, 1, 4, 4, 1, 1, 2, 3, 3, 2, 4, 2, 2, 0, 3, 0, 2, 1, 4, 2, 2, 3, 1, 2, 4, 2, 1, 2, 4, 0]
    return process_scrambled(labels, [os.path.join(module, 'data/eric1.txt')], sample_rate=250, **kwargs)

def digits_5_2_dataset(**kwargs):
    labels = [1, 4, 4, 3, 1, 4, 1, 4, 3, 0, 4, 3, 2, 3, 2, 1, 3, 1, 0, 1, 4, 0, 1, 3, 2, 2, 4, 4, 0, 4, 4, 0, 1, 4, 3, 4, 0, 0, 0, 4, 4, 0, 1, 2, 0, 2, 0, 0, 0, 0, 4, 1, 2, 2, 3, 2, 2, 0, 0, 2, 3, 3, 3, 3, 0, 3, 0, 3, 2, 1, 1, 4, 3, 3, 3, 1, 1, 4, 3, 4, 1, 0, 3, 0, 1, 4, 1, 2, 1, 1, 2, 3, 1, 2, 0, 0, 0, 3, 2, 0, 4, 1, 1, 4, 2, 4, 1, 4, 1, 4, 2, 2, 2, 1, 1, 3, 2, 4, 1, 3, 0, 3, 4, 3, 4, 3, 4, 3, 1, 2, 3, 3, 0, 4, 3, 0, 2, 2, 0, 4, 4, 3, 4, 1, 3, 4, 4, 0, 3, 0, 4, 3, 1, 2, 1, 3, 0, 1, 1, 3, 1, 2, 3, 0, 3, 0, 2, 3, 1, 3, 3, 3, 2, 2, 0, 2, 0, 4, 2, 3, 3, 2, 2, 4, 1, 0, 0, 4, 2, 1, 1, 0, 1, 0, 0, 0, 4, 0, 2, 3, 2, 4, 0, 4, 2, 3, 2, 4, 0, 4, 1, 2, 0, 1, 0, 1, 1, 0, 4, 2, 1, 1, 4, 4, 1, 1, 2, 3, 3, 2, 4, 2, 2, 0, 3, 0, 2, 1, 4, 2, 2, 3, 1, 2, 4, 2, 1, 2, 4, 0]
    return process_scrambled(labels, [os.path.join(module, 'data/eric2.txt')], sample_rate=250, **kwargs)

def digits_5_3_dataset(**kwargs):
    labels = [1, 4, 4, 3, 1, 4, 1, 4, 3, 0, 4, 3, 2, 3, 2, 1, 3, 1, 0, 1, 4, 0, 1, 3, 2, 2, 4, 4, 0, 4, 4, 0, 1, 4, 3, 4, 0, 0, 0, 4, 4, 0, 1, 2, 0, 2, 0, 0, 0, 0, 4, 1, 2, 2, 3, 2, 2, 0, 0, 2, 3, 3, 3, 3, 0, 3, 0, 3, 2, 1, 1, 4, 3, 3, 3, 1, 1, 4, 3, 4, 1, 0, 3, 0, 1, 4, 1, 2, 1, 1, 2, 3, 1, 2, 0, 0, 0, 3, 2, 0, 4, 1, 1, 4, 2, 4, 1, 4, 1, 4, 2, 2, 2, 1, 1, 3, 2, 4, 1, 3, 0, 3, 4, 3, 4, 3, 4, 3, 1, 2, 3, 3, 0, 4, 3, 0, 2, 2, 0, 4, 4, 3, 4, 1, 3, 4, 4, 0, 3, 0, 4, 3, 1, 2, 1, 3, 0, 1, 1, 3, 1, 2, 3, 0, 3, 0, 2, 3, 1, 3, 3, 3, 2, 2, 0, 2, 0, 4, 2, 3, 3, 2, 2, 4, 1, 0, 0, 4, 2, 1, 1, 0, 1, 0, 0, 0, 4, 0, 2, 3, 2, 4, 0, 4, 2, 3, 2, 4, 0, 4, 1, 2, 0, 1, 0, 1, 1, 0, 4, 2, 1, 1, 4, 4, 1, 1, 2, 3, 3, 2, 4, 2, 2, 0, 3, 0, 2, 1, 4, 2, 2, 3, 1, 2, 4, 2, 1, 2, 4, 0]
    return process_scrambled(labels, [os.path.join(module, 'data/eric3.txt')], sample_rate=250, **kwargs)

def digits_5_4_dataset(**kwargs):
    labels = [1, 4, 4, 3, 1, 4, 1, 4, 3, 0, 4, 3, 2, 3, 2, 1, 3, 1, 0, 1, 4, 0, 1, 3, 2, 2, 4, 4, 0, 4, 4, 0, 1, 4, 3, 4, 0, 0, 0, 4, 4, 0, 1, 2, 0, 2, 0, 0, 0, 0, 4, 1, 2, 2, 3, 2, 2, 0, 0, 2, 3, 3, 3, 3, 0, 3, 0, 3, 2, 1, 1, 4, 3, 3, 3, 1, 1, 4, 3, 4, 1, 0, 3, 0, 1, 4, 1, 2, 1, 1, 2, 3, 1, 2, 0, 0, 0, 3, 2, 0, 4, 1, 1, 4, 2, 4, 1, 4, 1, 4, 2, 2, 2, 1, 1, 3, 2, 4, 1, 3, 0, 3, 4, 3, 4, 3, 4, 3, 1, 2, 3, 3, 0, 4, 3, 0, 2, 2, 0, 4, 4, 3, 4, 1, 3, 4, 4, 0, 3, 0, 4, 3, 1, 2, 1, 3, 0, 1, 1, 3, 1, 2, 3, 0, 3, 0, 2, 3, 1, 3, 3, 3, 2, 2, 0, 2, 0, 4, 2, 3, 3, 2, 2, 4, 1, 0, 0, 4, 2, 1, 1, 0, 1, 0, 0, 0, 4, 0, 2, 3, 2, 4, 0, 4, 2, 3, 2, 4, 0, 4, 1, 2, 0, 1, 0, 1, 1, 0, 4, 2, 1, 1, 4, 4, 1, 1, 2, 3, 3, 2, 4, 2, 2, 0, 3, 0, 2, 1, 4, 2, 2, 3, 1, 2, 4, 2, 1, 2, 4, 0]
    return process_scrambled(labels, [os.path.join(module, 'data/eric4.txt')], sample_rate=250, **kwargs)

def digits_dataset(**kwargs):
    return combine([
#            digits_session_0_dataset(**kwargs),
#            digits_session_1_dataset(**kwargs),
##            digits_session_2_dataset(**kwargs),
#            digits_session_3_dataset(**kwargs),
#            digits_session_4_dataset(**kwargs),
#            digits_session_5_dataset(**kwargs),
#            digits_session_6_dataset(**kwargs),
#            digits_session_7_dataset(**kwargs),
#            digits_session_8_dataset(**kwargs),
#            digits_session_10_dataset(**kwargs),
#            digits_session_11_dataset(**kwargs),
#            digits_session_12_dataset(**kwargs),
#            digits_session_13_dataset(**kwargs),
#            digits_session_14_dataset(**kwargs),
#            digits_session_15_dataset(**kwargs),
#            digits_sequences_session_1_dataset(**kwargs),
            digits_sequences_session_2_dataset(**kwargs),
        ])

def words_16_dataset(**kwargs):
    return join([
            process(1, [os.path.join(module, 'data/20_subvocal_the_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/21_subvocal_a_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/23_subvocal_is_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/24_subvocal_it_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/25_subvocal_what_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/26_subvocal_where_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/27_subvocal_time_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/29_subvocal_year_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/30_subvocal_day_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/31_subvocal_plus_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/32_subvocal_minus_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/33_subvocal_about_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/34_subvocal_student_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/35_subvocal_government_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/36_subvocal_important_50_trials.txt')], **kwargs),
            process(1, [os.path.join(module, 'data/37_subvocal_information_50_trials.txt')], **kwargs),
        ])

def silence_dataset(**kwargs):
    kwargs['include_surrounding'] = False
    return [flatten([
#            flatten(process(1, [os.path.join(module, 'data/39_subvocal_silence_100_trials.txt')], **kwargs)),
#            flatten(process(1, [os.path.join(module, 'data/41_subvocal_silence_100_trials.txt')], **kwargs)),
#            flatten(process(1, [os.path.join(module, 'data/42_subvocal_silence_120_trials.txt')], **kwargs)),
#            flatten(process(1, [os.path.join(module, 'data/43_subvocal_silence_20_trials.txt')], **kwargs)), #s1
#            flatten(process(1, [os.path.join(module, 'data/44_subvocal_silence_104_trials.txt')], **kwargs)), #s1
#            flatten(process(1, [os.path.join(module, 'data/56_subvocal_silence_100_trials.txt')], **kwargs)), #s1
#            flatten(process(1, [os.path.join(module, 'data/57_subvocal_silence_217_trials.txt')], **kwargs)), #  s2
#            flatten(process(1, [os.path.join(module, 'data/68_subvocal_silence_196_trials.txt')], **kwargs)), #  s2
#            flatten(process(1, [os.path.join(module, 'data/74_subvocal_silence_300_trials.txt')], **kwargs)), #s3
#            flatten(process(1, [os.path.join(module, 'data/95_subvocal_silence_300_trials.txt')], **kwargs)), #  s5
#            flatten(process(1, [os.path.join(module, 'data/96_subvocal_silence_100_trials.txt')], **kwargs)), #  s5
#            flatten(process(1, [os.path.join(module, 'data/107_subvocal_silence_300_trials.txt')], **kwargs)), #s6
#            flatten(process(1, [os.path.join(module, 'data/108_subvocal_silence_100_trials.txt')], **kwargs)), #s6
#            flatten(process(1, [os.path.join(module, 'data/119_subvocal_silence_300_trials.txt')], **kwargs)), # s7
#            flatten(process(1, [os.path.join(module, 'data/123_subvocal_silence_100_trials.txt')], **kwargs)), #s8
#            flatten(process(1, [os.path.join(module, 'data/127_subvocal_silence_200_trials.txt')], **kwargs)), # s9
#            flatten(process(1, [os.path.join(module, 'data/128_subvocal_silence_300_trials.txt')], **kwargs)), #s10
#            flatten(process(1, [os.path.join(module, 'data/132_subvocal_silence_100_trials.txt')], **kwargs)), #s10
#            flatten(process(1, [os.path.join(module, 'data/139_subvocal_silence_100_trials.txt')], **kwargs)), #s10
#            flatten(process(1, [os.path.join(module, 'data/143_subvocal_silence_100_trials.txt')], **kwargs)), #s10
#            flatten(process(1, [os.path.join(module, 'data/150_subvocal_silence_200_trials.txt')], **kwargs)), #s14
#            flatten(process(1, [os.path.join(module, 'data/152_subvocal_silence_30_trials.txt')], **kwargs)), #s14
            flatten(process(1, [os.path.join(module, 'data/156_subvocal_silence_200_trials.txt')], **kwargs)), #s15
        ])]

def digits_and_silence_dataset(**kwargs):
    return join([
            digits_dataset(**kwargs),
            silence_dataset(**kwargs),
        ])

def words_16_and_silence_dataset(**kwargs):
    return join([
            words_16_dataset(**kwargs),
            silence_dataset(**kwargs),
        ])

def not_silence_dataset(**kwargs):
    return [flatten([
            flatten(digits_dataset(**kwargs)),
#            flatten(words_16_dataset(**kwargs)),
        ])]

def silence_and_not_silence_dataset(**kwargs):
    return join([
            silence_dataset(**kwargs),
            not_silence_dataset(**kwargs),
        ])
