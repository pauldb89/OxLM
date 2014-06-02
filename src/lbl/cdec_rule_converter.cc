#include "lbl/cdec_rule_converter.h"

namespace oxlm {

CdecRuleConverter::CdecRuleConverter(
    const boost::shared_ptr<CdecLBLMapper>& mapper,
    const boost::shared_ptr<CdecStateConverter>& state_converter)
    : mapper(mapper), stateConverter(state_converter) {}

vector<int> CdecRuleConverter::convertTargetSide(
    const vector<int>& target, const vector<const void*>& prev_states) const {
  vector<int> ret;
  for (int symbol: target) {
    if (symbol <= 0) {
      const auto& symbols = stateConverter->convert(prev_states[-symbol]);
      ret.insert(ret.end(), symbols.begin(), symbols.end());
    } else {
      ret.push_back(mapper->convert(symbol));
    }
  }
  return ret;
}

} // namespace oxlm
