from lib.ENSEMBLE import Ensemble
import warnings
warnings.filterwarnings('ignore')

ensemble = Ensemble(stdNum=1)
# ensemble.show('red', ['日期','時間','rank','MD','LOF','ISF'])
print(ensemble.metric())
