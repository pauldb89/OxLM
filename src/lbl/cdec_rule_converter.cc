#include <cassert>
#include "lbl/cdec_rule_converter.h"

namespace oxlm {

CdecRuleConverter::CdecRuleConverter(
    const boost::shared_ptr<CdecLBLMapper>& mapper,
    const boost::shared_ptr<CdecStateConverterBase>& state_converter)
    : mapper(mapper), stateConverter(state_converter) {}

// target is the target side of a hyperedge rule, encoded using cdec's vocab.
// prev_states will contain one state per non-terminal in target.
// Returns an encoding of the hyperedge using OxLM's vocabulary.
vector<int> CdecRuleConverter::convertTargetSide(
    const vector<int>& target, const vector<const void*>& prev_states) const {
  vector<int> ret;
  for (int symbol: target) {
    // Non-terminals are indexed starting at 0 and counting down
    // X_1 = 0, X_2 = -1, X_3 = -2, ...

    // If we see a terminal, map it from cdec's vocab to OxLM s vocab
    // If we have a non-terminal, it will have a corresponding element in prev_states.
    // Each state is just a sequence of symbols, so we load them and add them to the output. 
    if (symbol <= 0) {
      const auto& symbols = stateConverter->getTerminals(prev_states[-symbol]);
      ret.insert(ret.end(), symbols.begin(), symbols.end());
    } else {
      ret.push_back(mapper->convert(symbol));
    }
  }
  return ret;
}

} // namespace oxlm
