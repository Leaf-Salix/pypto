/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/codegen/orchestration/orchestration_analysis.h"
#include "pypto/codegen/orchestration_op_registry.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

using transform_utils::FlattenToStmts;

namespace {

constexpr const char* kStaticWindowArrayNameAttr = "static_window_array_name";
constexpr const char* kStaticWindowShapeAttr = "static_window_shape";
constexpr const char* kStaticWindowOffsetAttr = "static_window_offset";
constexpr const char* kStaticWindowLoopVarsAttr = "static_window_loop_vars";
constexpr const char* kStaticWindowLoopStartsAttr = "static_window_loop_starts";
constexpr const char* kStaticWindowLoopStepsAttr = "static_window_loop_steps";
constexpr const char* kStaticWindowLoopTripsAttr = "static_window_loop_trips";

ExprPtr SubstituteStaticWindowLoopValues(const ExprPtr& expr, const std::vector<VarPtr>& loop_vars,
                                         const std::vector<ExprPtr>& loop_starts,
                                         const std::vector<ExprPtr>& loop_steps,
                                         const std::vector<int64_t>& loop_trips, int64_t linear_index) {
  std::unordered_map<const Var*, ExprPtr> subst;
  int64_t remainder = linear_index;
  for (auto i = static_cast<int64_t>(loop_vars.size()) - 1; i >= 0; --i) {
    int64_t trip = loop_trips[static_cast<size_t>(i)];
    INTERNAL_CHECK(trip > 0) << "static window loop trip count must be positive";
    int64_t ordinal = remainder % trip;
    remainder /= trip;
    auto start = As<ConstInt>(loop_starts[static_cast<size_t>(i)]);
    auto step = As<ConstInt>(loop_steps[static_cast<size_t>(i)]);
    INTERNAL_CHECK(start && step) << "static window loop start/step must be const";
    int64_t value = start->value_ + ordinal * step->value_;
    subst[loop_vars[static_cast<size_t>(i)].get()] =
        std::make_shared<ConstInt>(value, DataType::INDEX, loop_vars[static_cast<size_t>(i)]->span_);
  }
  return arith::Analyzer().Simplify(transform_utils::Substitute(expr, subst));
}

std::string EmitStaticWindowAsUint32(const ExprPtr& expr, codegen::CodegenBase& codegen) {
  return "static_cast<uint32_t>(" + codegen.GenerateExprString(expr) + ")";
}

bool ExprUsesAnyVar(const ExprPtr& expr, const std::vector<VarPtr>& vars) {
  if (!expr || vars.empty()) return false;
  var_collectors::VarDefUseCollector collector;
  collector.VisitExpr(expr);
  for (const auto* var : collector.var_uses) {
    for (const auto& candidate : vars) {
      if (var == candidate.get()) return true;
    }
  }
  return false;
}

REGISTER_ORCHESTRATION_OP(tensor_precompute_static_windows, ("tensor.precompute_static_windows")) {
  CHECK(op->args_.size() == 1) << "tensor.precompute_static_windows expects parent tensor arg";
  auto parent_name = codegen.TryGetVarName(op->args_[0]);
  CHECK(!parent_name.empty()) << "tensor.precompute_static_windows parent must be a variable";
  parent_name = codegen.GetExternalTensorName(parent_name);

  auto array_name = op->GetAttr<std::string>(kStaticWindowArrayNameAttr, "");
  auto shape = op->GetAttr<std::vector<ExprPtr>>(kStaticWindowShapeAttr, {});
  auto offset = op->GetAttr<std::vector<ExprPtr>>(kStaticWindowOffsetAttr, {});
  auto loop_vars = op->GetAttr<std::vector<VarPtr>>(kStaticWindowLoopVarsAttr, {});
  auto loop_starts = op->GetAttr<std::vector<ExprPtr>>(kStaticWindowLoopStartsAttr, {});
  auto loop_steps = op->GetAttr<std::vector<ExprPtr>>(kStaticWindowLoopStepsAttr, {});
  auto loop_trips = op->GetAttr<std::vector<int64_t>>(kStaticWindowLoopTripsAttr, {});
  CHECK(!array_name.empty()) << "tensor.precompute_static_windows missing array name";
  CHECK(shape.size() == offset.size()) << "tensor.precompute_static_windows shape/offset rank mismatch";
  CHECK(loop_vars.size() == loop_starts.size() && loop_vars.size() == loop_steps.size() &&
        loop_vars.size() == loop_trips.size())
      << "tensor.precompute_static_windows loop metadata mismatch";

  int64_t window_count = 1;
  for (auto trip : loop_trips) window_count *= trip;
  CHECK(window_count > 0) << "tensor.precompute_static_windows requires at least one window";
  bool shape_is_loop_invariant = true;
  for (const auto& dim : shape) {
    if (ExprUsesAnyVar(dim, loop_vars)) {
      shape_is_loop_invariant = false;
      break;
    }
  }

  std::ostringstream oss;
  const size_t ndim = shape.size();

  oss << "uint32_t " << array_name << "_offsets[" << window_count << "][" << ndim << "];\n";
  oss << "for (int64_t " << array_name << "_i = 0; " << array_name << "_i < " << window_count << "; ++"
      << array_name << "_i) {";
  if (!loop_vars.empty()) {
    oss << "\n    int64_t " << array_name << "_rem = " << array_name << "_i;";
    for (auto i = static_cast<int64_t>(loop_vars.size()) - 1; i >= 0; --i) {
      const auto idx = static_cast<size_t>(i);
      auto start = As<ConstInt>(loop_starts[idx]);
      auto step = As<ConstInt>(loop_steps[idx]);
      INTERNAL_CHECK(start && step) << "static window loop start/step must be const";
      oss << "\n    int64_t " << codegen.GenerateExprString(loop_vars[idx]) << " = " << start->value_
          << " + (" << array_name << "_rem % " << loop_trips[idx] << ") * " << step->value_ << ";";
      if (i > 0) {
        oss << "\n    " << array_name << "_rem /= " << loop_trips[idx] << ";";
      }
    }
  }
  for (size_t d = 0; d < ndim; ++d) {
    oss << "\n    " << array_name << "_offsets[" << array_name << "_i][" << d
        << "] = " << EmitStaticWindowAsUint32(offset[d], codegen) << ";";
  }
  oss << "\n}\n";

  if (shape_is_loop_invariant) {
    oss << "uint32_t " << array_name << "_shapes[" << window_count << "][" << ndim << "];\n";
    oss << "for (int64_t " << array_name << "_i = 0; " << array_name << "_i < " << window_count << "; ++"
        << array_name << "_i) {";
    for (size_t d = 0; d < ndim; ++d) {
      std::string shape_index = array_name + "_i";
      std::string offset_slot = array_name + "_offsets[" + shape_index + "][" + std::to_string(d) + "]";
      oss << "\n    " << array_name << "_shapes[" << shape_index << "][" << d << "] = ";
      oss << "(" << offset_slot << " >= " << parent_name << ".shapes[" << d << "] ? 0u : std::min<uint32_t>(";
      oss << EmitStaticWindowAsUint32(shape[d], codegen) << ", " << parent_name << ".shapes[" << d << "] - "
          << offset_slot << "));";
    }
    oss << "\n}\n";
  } else {
    oss << "uint32_t " << array_name << "_shapes[" << window_count << "][" << ndim << "];\n";
    oss << "for (int64_t " << array_name << "_i = 0; " << array_name << "_i < " << window_count << "; ++"
        << array_name << "_i) {";
    if (!loop_vars.empty()) {
      oss << "\n    int64_t " << array_name << "_rem = " << array_name << "_i;";
      for (auto i = static_cast<int64_t>(loop_vars.size()) - 1; i >= 0; --i) {
        const auto idx = static_cast<size_t>(i);
        auto start = As<ConstInt>(loop_starts[idx]);
        auto step = As<ConstInt>(loop_steps[idx]);
        INTERNAL_CHECK(start && step) << "static window loop start/step must be const";
        oss << "\n    int64_t " << codegen.GenerateExprString(loop_vars[idx]) << " = " << start->value_
            << " + (" << array_name << "_rem % " << loop_trips[idx] << ") * " << step->value_ << ";";
        if (i > 0) {
          oss << "\n    " << array_name << "_rem /= " << loop_trips[idx] << ";";
        }
      }
    }
    for (size_t d = 0; d < ndim; ++d) {
      std::string shape_index = array_name + "_i";
      std::string offset_slot = array_name + "_offsets[" + shape_index + "][" + std::to_string(d) + "]";
      oss << "\n    " << array_name << "_shapes[" << shape_index << "][" << d << "] = ";
      oss << "(" << offset_slot << " >= " << parent_name << ".shapes[" << d << "] ? 0u : std::min<uint32_t>(";
      oss << EmitStaticWindowAsUint32(shape[d], codegen) << ", " << parent_name << ".shapes[" << d << "] - "
          << offset_slot << "));";
    }
    oss << "\n}\n";
  }

  oss << "auto " << array_name << "_make = [&](int64_t i) { return " << parent_name << ".view(" << array_name
      << "_shapes[i], " << array_name << "_offsets[i]); };\n";
  oss << "Tensor " << array_name << "[" << window_count << "] = {";
  constexpr int64_t kWindowsPerLine = 8;
  for (int64_t i = 0; i < window_count; ++i) {
    if (i % kWindowsPerLine == 0) {
      oss << "\n    ";
    } else {
      oss << " ";
    }
    oss << array_name << "_make(" << i << ")";
    if (i + 1 < window_count) oss << ",";
  }
  oss << "\n};";
  return oss.str();
}

REGISTER_ORCHESTRATION_OP(tensor_static_window_get, ("tensor.static_window_get")) {
  auto array_name = op->GetAttr<std::string>(kStaticWindowArrayNameAttr, "");
  auto loop_vars = op->GetAttr<std::vector<VarPtr>>(kStaticWindowLoopVarsAttr, {});
  auto loop_starts = op->GetAttr<std::vector<ExprPtr>>(kStaticWindowLoopStartsAttr, {});
  auto loop_steps = op->GetAttr<std::vector<ExprPtr>>(kStaticWindowLoopStepsAttr, {});
  auto loop_trips = op->GetAttr<std::vector<int64_t>>(kStaticWindowLoopTripsAttr, {});
  CHECK(!array_name.empty()) << "tensor.static_window_get missing array name";
  CHECK(loop_vars.size() == loop_starts.size() && loop_vars.size() == loop_steps.size() &&
        loop_vars.size() == loop_trips.size())
      << "tensor.static_window_get loop metadata mismatch";

  std::string idx_expr = "0";
  for (size_t i = 0; i < loop_vars.size(); ++i) {
    std::string loop_name = codegen.GenerateExprString(loop_vars[i]);
    std::string start = codegen.GenerateExprString(loop_starts[i]);
    std::string step = codegen.GenerateExprString(loop_steps[i]);
    std::string loop_idx =
        (start == "0" && step == "1") ? loop_name : "((" + loop_name + " - " + start + ") / " + step + ")";
    idx_expr =
        (i == 0) ? loop_idx : "(" + idx_expr + " * " + std::to_string(loop_trips[i]) + " + " + loop_idx + ")";
  }

  std::string result = codegen.GetCurrentResultTarget();
  std::ostringstream oss;
  oss << "int64_t " << result << "__window_idx = " << idx_expr << ";\n";
  oss << "const Tensor& " << result << " = " << array_name << "[" << result << "__window_idx];";
  return oss.str();
}

// ============================================================================
// Shared helpers
// ============================================================================

/// Find a function by name in a program.
FunctionPtr FindFunction(const ProgramPtr& program, const std::string& name) {
  for (const auto& [gvar, func] : program->functions_) {
    if (func->name_ == name) return func;
  }
  return nullptr;
}

/// Get the GlobalVar name from a Call, or empty string.
std::string GetCallFuncName(const CallPtr& call) {
  auto gvar = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
  return gvar ? gvar->name_ : "";
}

/// Compute row-major strides from a shape: [D1*D2*...*Dn, D2*...*Dn, ..., 1].
/// Returns empty vector if any dimension is not a ConstInt.
std::vector<ExprPtr> ComputeRowMajorStrides(const std::vector<ExprPtr>& shape) {
  std::vector<int64_t> dims;
  dims.reserve(shape.size());
  for (const auto& dim : shape) {
    auto ci = As<ConstInt>(dim);
    if (!ci) return {};
    dims.push_back(ci->value_);
  }

  size_t ndim = dims.size();
  std::vector<ExprPtr> strides(ndim);
  int64_t product = 1;
  for (size_t i = ndim; i > 0; --i) {
    strides[i - 1] = std::make_shared<ConstInt>(product, DataType::INDEX, Span::unknown());
    product *= dims[i - 1];
  }
  return strides;
}

std::string MakeUniqueFunctionName(const ProgramPtr& program, const std::string& base_name) {
  if (!program || !program->GetFunction(base_name)) return base_name;
  for (size_t suffix = 1;; ++suffix) {
    auto candidate = base_name + "_" + std::to_string(suffix);
    if (!program->GetFunction(candidate)) return candidate;
  }
}

/// Count Var/IterArg references to `target` inside a statement subtree.
size_t CountVarRefsInStmt(const StmtPtr& stmt, const Var* target) {
  class Counter : public IRVisitor {
   public:
    explicit Counter(const Var* target) : target_(target) {}

    [[nodiscard]] size_t count() const { return count_; }

   protected:
    void VisitExpr_(const VarPtr& op) override {
      if (op.get() == target_) ++count_;
      IRVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const IterArgPtr& op) override {
      if (op.get() == target_) ++count_;
      IRVisitor::VisitExpr_(op);
    }

   private:
    const Var* target_;
    size_t count_ = 0;
  };

  Counter counter(target);
  counter.VisitStmt(stmt);
  return counter.count();
}

bool ExprReferencesOnlyVarsIn(const ExprPtr& expr, const std::unordered_set<const Var*>& allowed) {
  class Checker : public IRVisitor {
   public:
    explicit Checker(const std::unordered_set<const Var*>& allowed) : allowed_(allowed) {}

    [[nodiscard]] bool ok() const { return ok_; }

   protected:
    void VisitExpr_(const VarPtr& op) override {
      if (!allowed_.count(op.get())) ok_ = false;
    }

    void VisitExpr_(const IterArgPtr& op) override {
      if (!allowed_.count(op.get())) ok_ = false;
    }

   private:
    const std::unordered_set<const Var*>& allowed_;
    bool ok_ = true;
  };

  Checker checker(allowed);
  checker.VisitExpr(expr);
  return checker.ok();
}

bool IsAllZeroOffsets(const std::vector<ExprPtr>& offsets) {
  for (const auto& offset : offsets) {
    auto ci = As<ConstInt>(offset);
    if (!ci || ci->value_ != 0) return false;
  }
  return true;
}

bool ContainsGeneratedChunkLoop(const FunctionPtr& func) {
  class Finder : public IRVisitor {
   public:
    [[nodiscard]] bool found() const { return found_; }

   protected:
    void VisitStmt_(const ForStmtPtr& op) override {
      if (op->GetAttr<LoopOrigin>("loop_origin", LoopOrigin::Original) != LoopOrigin::Original) {
        found_ = true;
        return;
      }
      IRVisitor::VisitStmt_(op);
    }

   private:
    bool found_ = false;
  };

  if (!func) return false;
  Finder finder;
  finder.VisitStmt(func->body_);
  return finder.found();
}

bool IsTensorAllocationOp(const CallPtr& call) {
  if (!call || std::dynamic_pointer_cast<const GlobalVar>(call->op_)) return false;
  return call->op_->name_ == "tensor.create" || call->op_->name_ == "tensor.full";
}

std::unordered_set<const Var*> CollectLoopLocalTensorAllocs(const ForStmtPtr& loop) {
  class Collector : public IRVisitor {
   public:
    [[nodiscard]] const std::unordered_set<const Var*>& result() const { return result_; }

   protected:
    void VisitStmt_(const AssignStmtPtr& op) override {
      auto call = As<Call>(op->value_);
      if (IsTensorAllocationOp(call) && As<TensorType>(op->var_->GetType())) {
        result_.insert(op->var_.get());
      }
      IRVisitor::VisitStmt_(op);
    }

   private:
    std::unordered_set<const Var*> result_;
  };

  if (!loop) return {};
  Collector collector;
  collector.VisitStmt(loop->body_);
  return collector.result();
}

std::vector<size_t> CollectOutParamIndices(const FunctionPtr& func) {
  std::vector<size_t> result;
  if (!func) return result;
  for (size_t i = 0; i < func->param_directions_.size() && i < func->params_.size(); ++i) {
    if (func->param_directions_[i] == ParamDirection::Out) {
      result.push_back(i);
    }
  }
  return result;
}

bool IsTensorTypedArg(const ExprPtr& arg) {
  TypePtr ty = arg ? arg->GetType() : TypePtr{};
  if (!ty) return false;
  return As<TensorType>(ty) || As<TupleType>(ty);
}

/// Info about an InCore function's Out params and their return mappings.
struct OutParamReturnMapping {
  size_t param_index;   ///< Position in param list
  size_t return_index;  ///< Which return value stores to this Out param
  VarPtr param_var;     ///< The Out param variable
};

/// Build the mapping from Out params to return indices for an InCore function.
/// Scans tile.store calls before the ReturnStmt to find which Out param
/// each return value stores to.
std::vector<OutParamReturnMapping> BuildOutParamReturnMappings(const FunctionPtr& func) {
  // Collect Out param vars and their indices
  std::unordered_map<const Var*, size_t> out_var_to_param_idx;
  for (size_t i = 0; i < func->params_.size(); ++i) {
    if (i < func->param_directions_.size() && func->param_directions_[i] == ParamDirection::Out) {
      out_var_to_param_idx[func->params_[i].get()] = i;
    }
  }
  if (out_var_to_param_idx.empty()) return {};

  auto body_stmts = FlattenToStmts(func->body_);

  // Build var->assign map for quick lookup
  std::unordered_map<const Var*, AssignStmtPtr> var_def;
  for (const auto& stmt : body_stmts) {
    if (auto assign = As<AssignStmt>(stmt)) {
      var_def[assign->var_.get()] = assign;
    }
  }

  // Find return statement
  ReturnStmtPtr return_stmt;
  for (const auto& stmt : body_stmts) {
    if (auto ret = As<ReturnStmt>(stmt)) {
      return_stmt = ret;
      break;
    }
  }
  if (!return_stmt) return {};

  std::vector<OutParamReturnMapping> result;

  for (size_t ret_i = 0; ret_i < return_stmt->value_.size(); ++ret_i) {
    auto ret_var = As<Var>(return_stmt->value_[ret_i]);
    if (!ret_var) continue;

    auto def_it = var_def.find(ret_var.get());
    if (def_it == var_def.end()) continue;

    auto call = As<Call>(def_it->second->value_);
    if (!call || call->op_->name_ != "tile.store") continue;

    if (call->args_.size() < 3) continue;
    auto out_tensor = As<Var>(call->args_[2]);
    if (!out_tensor) continue;

    auto param_it = out_var_to_param_idx.find(out_tensor.get());
    if (param_it == out_var_to_param_idx.end()) continue;

    result.push_back({param_it->second, ret_i, func->params_[param_it->second]});
  }

  return result;
}

// ============================================================================
// Pattern 1: IterArgReuseOptimizer
//
// Detects when a tensor.create for an InCore Out param is inside a
// ForStmt/WhileStmt loop where the InCore result feeds back as an iter-arg,
// and the corresponding In param receives the iter-arg value.
//
// Optimization: remove the tensor.create, remove the Out param from the InCore
// function, promote the In param to InOut, redirect tile.store to the In param.
// ============================================================================

class IterArgReuseOptimizer {
 public:
  ProgramPtr Run(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names) {
    auto reuse_results = Analyze(program, incore_names);

    // Rewrite InCore functions
    std::unordered_map<std::string, FunctionPtr> rewritten_incores;
    for (auto& [fname, reuse] : reuse_results) {
      auto func = FindFunction(program, fname);
      if (!func) continue;
      rewritten_incores[fname] = RewriteIncore(func, reuse.merges);
    }

    // Build the new function list
    std::vector<FunctionPtr> new_functions;
    for (const auto& [gvar, func] : program->functions_) {
      if (rewritten_incores.count(func->name_)) {
        new_functions.push_back(rewritten_incores[func->name_]);
      } else if (!incore_names.count(func->name_)) {
        // Orchestration function: rewrite call sites
        DeadCreateScanner scanner(reuse_results);
        scanner.VisitStmt(func->body_);

        CallSiteRewriter rewriter(reuse_results, rewritten_incores, scanner.dead_creates());
        auto new_body = rewriter.VisitStmt(func->body_);
        if (new_body.get() != func->body_.get()) {
          auto new_func = MutableCopy(func);
          new_func->body_ = new_body;
          new_functions.push_back(new_func);
        } else {
          new_functions.push_back(func);
        }
      } else {
        new_functions.push_back(func);
      }
    }

    return std::make_shared<Program>(new_functions, program->name_, program->span_);
  }

 private:
  /// A single Out->In merge for an InCore function.
  struct OutToInMerge {
    size_t out_param_index;
    size_t in_param_index;
  };

  /// Per-InCore-function analysis result.
  struct AnalysisResult {
    std::string func_name;
    std::vector<OutToInMerge> merges;
  };

  // -- Analysis: IRVisitor that finds ForStmt/WhileStmt with iter-arg reuse --

  class LoopAnalyzer : public IRVisitor {
   public:
    LoopAnalyzer(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names,
                 const std::unordered_map<std::string, std::vector<OutParamReturnMapping>>& out_mappings)
        : program_(program), incore_names_(incore_names), out_mappings_(out_mappings) {}

    const std::unordered_map<std::string, AnalysisResult>& results() const { return results_; }

   protected:
    void VisitStmt_(const ForStmtPtr& op) override {
      IRVisitor::VisitStmt_(op);  // Recurse into body first
      if (!op->iter_args_.empty()) {
        AnalyzeLoop(op->iter_args_, op->body_);
      }
    }

    void VisitStmt_(const WhileStmtPtr& op) override {
      IRVisitor::VisitStmt_(op);
      if (!op->iter_args_.empty()) {
        AnalyzeLoop(op->iter_args_, op->body_);
      }
    }

   private:
    void AnalyzeLoop(const std::vector<IterArgPtr>& iter_args, const StmtPtr& body) {
      auto loop_body_stmts = FlattenToStmts(body);

      // Collect tensor.create vars and InCore call assignments
      std::unordered_set<const Var*> tensor_create_vars;
      std::vector<AssignStmtPtr> incore_calls;

      for (const auto& stmt : loop_body_stmts) {
        auto assign = As<AssignStmt>(stmt);
        if (!assign) continue;
        auto call = As<Call>(assign->value_);
        if (!call) continue;

        if (!std::dynamic_pointer_cast<const GlobalVar>(call->op_) && call->op_->name_ == "tensor.create") {
          tensor_create_vars.insert(assign->var_.get());
        } else {
          auto fname = GetCallFuncName(call);
          if (incore_names_.count(fname)) {
            incore_calls.push_back(assign);
          }
        }
      }

      // Find yield statement
      auto yield = transform_utils::FindYieldStmt(body);
      if (!yield) return;

      // Build tuple extract map: call_result_var -> {tuple_index -> dest_var}
      std::unordered_map<const Var*, std::unordered_map<size_t, const Var*>> tuple_extracts;
      for (const auto& stmt : loop_body_stmts) {
        auto assign = As<AssignStmt>(stmt);
        if (!assign) continue;
        auto tgi = As<TupleGetItemExpr>(assign->value_);
        if (!tgi) continue;
        if (auto src_var = As<Var>(tgi->tuple_)) {
          tuple_extracts[src_var.get()][static_cast<size_t>(tgi->index_)] = assign->var_.get();
        }
      }

      // Analyze each InCore call
      for (const auto& call_assign : incore_calls) {
        auto call = As<Call>(call_assign->value_);
        auto fname = GetCallFuncName(call);
        auto mapping_it = out_mappings_.find(fname);
        if (mapping_it == out_mappings_.end()) continue;
        const auto& out_param_mappings = mapping_it->second;

        auto incore_func = FindFunction(program_, fname);
        if (!incore_func) continue;

        AnalysisResult reuse_result;
        reuse_result.func_name = fname;

        for (const auto& opm : out_param_mappings) {
          if (opm.param_index >= call->args_.size()) continue;
          auto out_arg_var = As<Var>(call->args_[opm.param_index]);
          if (!out_arg_var || !tensor_create_vars.count(out_arg_var.get())) continue;

          // Trace return[ret_i] -> yield[y_i]
          size_t yield_index = SIZE_MAX;

          if (out_param_mappings.size() == 1 && incore_func->return_types_.size() == 1) {
            for (size_t yi = 0; yi < yield->value_.size(); ++yi) {
              auto yv = As<Var>(yield->value_[yi]);
              if (yv && yv.get() == call_assign->var_.get()) {
                yield_index = yi;
                break;
              }
            }
          } else {
            auto extract_it = tuple_extracts.find(call_assign->var_.get());
            if (extract_it != tuple_extracts.end()) {
              auto ret_it = extract_it->second.find(opm.return_index);
              if (ret_it != extract_it->second.end()) {
                for (size_t yi = 0; yi < yield->value_.size(); ++yi) {
                  auto yv = As<Var>(yield->value_[yi]);
                  if (yv && yv.get() == ret_it->second) {
                    yield_index = yi;
                    break;
                  }
                }
              }
            }
          }
          if (yield_index == SIZE_MAX) continue;

          // Check iter_arg[yield_index] maps to an In param of the same call
          if (yield_index >= iter_args.size()) continue;
          const auto* iter_arg_ptr = iter_args[yield_index].get();

          for (size_t arg_i = 0; arg_i < call->args_.size(); ++arg_i) {
            const Var* raw_ptr = nullptr;
            if (auto var = As<Var>(call->args_[arg_i])) {
              raw_ptr = var.get();
            } else if (auto ia = As<IterArg>(call->args_[arg_i])) {
              raw_ptr = ia.get();
            }
            if (raw_ptr != iter_arg_ptr) continue;

            if (arg_i < incore_func->param_directions_.size() &&
                incore_func->param_directions_[arg_i] == ParamDirection::In) {
              reuse_result.merges.push_back({opm.param_index, arg_i});
            }
            break;
          }
        }

        if (!reuse_result.merges.empty()) {
          if (results_.find(fname) == results_.end()) {
            results_[fname] = std::move(reuse_result);
          }
        }
      }
    }

    const ProgramPtr& program_;
    const std::unordered_set<std::string>& incore_names_;
    const std::unordered_map<std::string, std::vector<OutParamReturnMapping>>& out_mappings_;
    std::unordered_map<std::string, AnalysisResult> results_;
  };

  // -- Analysis: InCore internal In↔Out param pairings ----------------------
  //
  // A callee's In param and Out param are aliasing-compatible when the callee
  // reads the In fully via `tile.load` and writes the Out fully via `tile.store`
  // of a value that flows through at least one loop iter_arg chain starting
  // from that tile.load. The loop hop is what signals the In/Out are meant to
  // share storage (an accumulator); a plain load→store pair does not.

  /// Check that a tile.load call reads the full tensor — all offsets zero and
  /// both `shapes` and `valid_shapes` match the tensor shape dimension-by-
  /// dimension. `valid_shapes` differs from `shapes` for masked/padded loads,
  /// which must NOT be treated as full loads.
  static bool IsFullTensorLoad(const CallPtr& load_call, const TensorTypePtr& tensor_type) {
    if (!load_call || load_call->args_.size() < 4 || !tensor_type) return false;
    auto offsets = As<MakeTuple>(load_call->args_[1]);
    auto load_shape = As<MakeTuple>(load_call->args_[2]);
    auto valid_shape = As<MakeTuple>(load_call->args_[3]);
    if (!offsets || !load_shape || !valid_shape) return false;
    const size_t ndim = tensor_type->shape_.size();
    if (offsets->elements_.size() != ndim || load_shape->elements_.size() != ndim ||
        valid_shape->elements_.size() != ndim) {
      return false;
    }
    for (size_t i = 0; i < ndim; ++i) {
      auto want = std::dynamic_pointer_cast<const ConstInt>(tensor_type->shape_[i]);
      auto got_load = std::dynamic_pointer_cast<const ConstInt>(load_shape->elements_[i]);
      auto got_valid = std::dynamic_pointer_cast<const ConstInt>(valid_shape->elements_[i]);
      if (!want || !got_load || !got_valid) return false;
      if (want->value_ != got_load->value_ || want->value_ != got_valid->value_) return false;
      if (!IsConstValue(offsets->elements_[i], 0)) return false;
    }
    return true;
  }

  /// Compare two TensorTypes for compatible constant shape + dtype.
  static bool TensorTypesMatch(const TypePtr& a, const TypePtr& b) {
    auto ta = As<TensorType>(a);
    auto tb = As<TensorType>(b);
    if (!ta || !tb || ta->dtype_ != tb->dtype_) return false;
    if (ta->shape_.size() != tb->shape_.size()) return false;
    for (size_t i = 0; i < ta->shape_.size(); ++i) {
      auto ca = std::dynamic_pointer_cast<const ConstInt>(ta->shape_[i]);
      auto cb = std::dynamic_pointer_cast<const ConstInt>(tb->shape_[i]);
      if (!ca || !cb || ca->value_ != cb->value_) return false;
    }
    return true;
  }

  /// Walk body collecting: AssignStmt var_def map, ForStmt/WhileStmt
  /// return_var → iter_arg init value map, and the top-level ReturnStmt.
  class IterChainCollector : public IRVisitor {
   public:
    std::unordered_map<const Var*, AssignStmtPtr> var_def;
    std::unordered_map<const Var*, ExprPtr> return_var_to_init;
    ReturnStmtPtr return_stmt;

   protected:
    void VisitStmt_(const AssignStmtPtr& op) override {
      var_def[op->var_.get()] = op;
      IRVisitor::VisitStmt_(op);
    }
    void VisitStmt_(const ReturnStmtPtr& op) override {
      if (!return_stmt) return_stmt = op;
      IRVisitor::VisitStmt_(op);
    }
    void VisitStmt_(const ForStmtPtr& op) override {
      for (size_t i = 0; i < op->return_vars_.size() && i < op->iter_args_.size(); ++i) {
        return_var_to_init[op->return_vars_[i].get()] = op->iter_args_[i]->initValue_;
      }
      IRVisitor::VisitStmt_(op);
    }
    void VisitStmt_(const WhileStmtPtr& op) override {
      for (size_t i = 0; i < op->return_vars_.size() && i < op->iter_args_.size(); ++i) {
        return_var_to_init[op->return_vars_[i].get()] = op->iter_args_[i]->initValue_;
      }
      IRVisitor::VisitStmt_(op);
    }
  };

  /// For each Out param, trace `tile.store` source back through loop iter_arg
  /// chains to a `tile.load` of an In param. Returns (in_idx, out_idx) pairs
  /// where the chain exists, types match, and the load covers the full tensor.
  static std::vector<std::pair<size_t, size_t>> BuildInOutParamPairings(const FunctionPtr& func) {
    std::vector<std::pair<size_t, size_t>> pairings;

    std::unordered_map<const Var*, size_t> in_param_idx;
    for (size_t i = 0; i < func->params_.size() && i < func->param_directions_.size(); ++i) {
      if (func->param_directions_[i] != ParamDirection::In) continue;
      if (!As<TensorType>(func->params_[i]->GetType())) continue;
      in_param_idx[func->params_[i].get()] = i;
    }
    auto out_mappings = BuildOutParamReturnMappings(func);
    if (in_param_idx.empty() || out_mappings.empty()) return pairings;

    IterChainCollector collector;
    collector.VisitStmt(func->body_);
    if (!collector.return_stmt) return pairings;

    std::unordered_set<size_t> used_in_indices;
    for (const auto& opm : out_mappings) {
      if (opm.return_index >= collector.return_stmt->value_.size()) continue;
      auto ret_var = As<Var>(collector.return_stmt->value_[opm.return_index]);
      if (!ret_var) continue;
      auto ret_def = collector.var_def.find(ret_var.get());
      if (ret_def == collector.var_def.end()) continue;
      auto store_call = As<Call>(ret_def->second->value_);
      if (!store_call || store_call->op_->name_ != "tile.store" || store_call->args_.empty()) continue;

      // Trace backward through iter_arg chains. Require at least one loop hop:
      // a bare tile.load → tile.store without an accumulator has no semantic
      // indication that In and Out were intended to alias.
      const Var* current = nullptr;
      if (auto src_var = As<Var>(store_call->args_[0])) current = src_var.get();
      int loop_hops = 0;
      for (int hops = 0; hops < 16 && current; ++hops) {
        auto it = collector.return_var_to_init.find(current);
        if (it == collector.return_var_to_init.end()) break;
        auto init_var = As<Var>(it->second);
        if (!init_var) {
          current = nullptr;
          break;
        }
        current = init_var.get();
        ++loop_hops;
      }
      if (!current || loop_hops == 0) continue;

      auto load_def = collector.var_def.find(current);
      if (load_def == collector.var_def.end()) continue;
      auto load_call = As<Call>(load_def->second->value_);
      if (!load_call || load_call->op_->name_ != "tile.load" || load_call->args_.empty()) continue;
      auto load_src = As<Var>(load_call->args_[0]);
      if (!load_src) continue;
      auto in_it = in_param_idx.find(load_src.get());
      if (in_it == in_param_idx.end()) continue;

      auto tensor_type = As<TensorType>(func->params_[in_it->second]->GetType());
      if (!TensorTypesMatch(tensor_type, opm.param_var->GetType())) continue;
      if (!IsFullTensorLoad(load_call, tensor_type)) continue;

      if (!used_in_indices.insert(in_it->second).second) continue;
      pairings.emplace_back(in_it->second, opm.param_index);
    }
    return pairings;
  }

  // -- Analysis: standalone (non-looped) InCore calls whose In/Out can merge -

  /// One-shot visitor that collects everything the standalone analyzer needs
  /// from an orchestration function body: per-Var use counts, the set of Vars
  /// assigned by `tensor.create`, and the expression AST of each AssignStmt.
  /// Keeping it all in a single walk keeps per-function analysis O(N).
  ///
  /// Counts exclude definitional occurrences (AssignStmt LHS, loop_var,
  /// return_vars, iter_arg self-refs) so `use_count[v]` is the number of
  /// real reads of `v` in expressions.
  class FunctionBodyIndex : public IRVisitor {
   public:
    std::unordered_map<const Var*, size_t> use_count;
    std::unordered_set<const Var*> local_creates;

   protected:
    void VisitExpr_(const VarPtr& op) override { ++use_count[op.get()]; }
    void VisitExpr_(const IterArgPtr& op) override { ++use_count[op.get()]; }

    void VisitStmt_(const AssignStmtPtr& op) override {
      // Skip LHS (a def); visit only the RHS value.
      VisitExpr(op->value_);
      if (auto call = As<Call>(op->value_); call && !std::dynamic_pointer_cast<const GlobalVar>(call->op_) &&
                                            call->op_->name_ == "tensor.create") {
        local_creates.insert(op->var_.get());
      }
    }

    void VisitStmt_(const ForStmtPtr& op) override {
      VisitExpr(op->start_);
      VisitExpr(op->stop_);
      VisitExpr(op->step_);
      for (const auto& ia : op->iter_args_) {
        if (ia->initValue_) VisitExpr(ia->initValue_);
      }
      VisitStmt(op->body_);
    }

    void VisitStmt_(const WhileStmtPtr& op) override {
      VisitExpr(op->condition_);
      for (const auto& ia : op->iter_args_) {
        if (ia->initValue_) VisitExpr(ia->initValue_);
      }
      VisitStmt(op->body_);
    }
  };

  /// Count references to `target` within a single expression tree.
  static size_t CountVarRefs(const ExprPtr& expr, const Var* target) {
    class Counter : public IRVisitor {
     public:
      const Var* target;
      size_t count = 0;
      void VisitExpr_(const VarPtr& op) override {
        if (op.get() == target) ++count;
      }
      void VisitExpr_(const IterArgPtr& op) override {
        if (op.get() == target) ++count;
      }
    } c;
    c.target = target;
    c.VisitExpr(expr);
    return c.count;
  }

  /// Record of a standalone InCore call site. Owns references to the call's
  /// orchestration-function context so that we can test each candidate merge
  /// against every call site of the same callee before recording it.
  struct StandaloneCallSite {
    const FunctionBodyIndex* body_index;
    AssignStmtPtr assign_stmt;
    CallPtr call;
  };

  /// Collects standalone InCore calls (those outside any iter-arg-carrying
  /// loop) in an orchestration function body, keyed by callee name.
  class StandaloneCallCollector : public IRVisitor {
   public:
    StandaloneCallCollector(const std::unordered_set<std::string>& incore_names,
                            const FunctionBodyIndex& body_index,
                            std::unordered_map<std::string, std::vector<StandaloneCallSite>>& out)
        : incore_names_(incore_names), body_index_(body_index), out_(out) {}

   protected:
    void VisitStmt_(const ForStmtPtr& op) override {
      bool prev = inside_iter_loop_;
      if (!op->iter_args_.empty()) inside_iter_loop_ = true;
      IRVisitor::VisitStmt_(op);
      inside_iter_loop_ = prev;
    }
    void VisitStmt_(const WhileStmtPtr& op) override {
      bool prev = inside_iter_loop_;
      if (!op->iter_args_.empty()) inside_iter_loop_ = true;
      IRVisitor::VisitStmt_(op);
      inside_iter_loop_ = prev;
    }
    void VisitStmt_(const AssignStmtPtr& op) override {
      if (!inside_iter_loop_) {
        if (auto call = As<Call>(op->value_)) {
          auto fname = GetCallFuncName(call);
          if (!fname.empty() && incore_names_.count(fname)) {
            out_[fname].push_back({&body_index_, op, call});
          }
        }
      }
      IRVisitor::VisitStmt_(op);
    }

   private:
    const std::unordered_set<std::string>& incore_names_;
    const FunctionBodyIndex& body_index_;
    std::unordered_map<std::string, std::vector<StandaloneCallSite>>& out_;
    bool inside_iter_loop_ = false;
  };

  /// Check whether a (in_idx, out_idx) pairing is safe to apply at `site`:
  /// the Out arg is a locally-allocated `tensor.create`, and the In arg's
  /// sole use in the enclosing orch function is this call.
  static bool IsPairingSafeAtCallSite(const StandaloneCallSite& site, size_t in_idx, size_t out_idx) {
    const auto& call = site.call;
    if (out_idx >= call->args_.size() || in_idx >= call->args_.size()) return false;
    auto out_var = As<Var>(call->args_[out_idx]);
    auto in_var = As<Var>(call->args_[in_idx]);
    if (!out_var || !in_var) return false;
    if (!site.body_index->local_creates.count(out_var.get())) return false;

    auto use_it = site.body_index->use_count.find(in_var.get());
    size_t total_refs = use_it == site.body_index->use_count.end() ? 0 : use_it->second;
    size_t self_refs = CountVarRefs(site.assign_stmt->value_, in_var.get());
    return total_refs == self_refs;
  }

  /// Analyze orchestration functions for iter-arg reuse opportunities.
  std::unordered_map<std::string, AnalysisResult> Analyze(
      const ProgramPtr& program, const std::unordered_set<std::string>& incore_names) {
    std::unordered_map<std::string, std::vector<OutParamReturnMapping>> out_mappings;
    std::unordered_map<std::string, std::vector<std::pair<size_t, size_t>>> in_out_pairings;
    for (const auto& [gvar, func] : program->functions_) {
      if (!incore_names.count(func->name_)) continue;
      out_mappings[func->name_] = BuildOutParamReturnMappings(func);
      in_out_pairings[func->name_] = BuildInOutParamPairings(func);
    }

    LoopAnalyzer analyzer(program, incore_names, out_mappings);
    for (const auto& [gvar, func] : program->functions_) {
      if (incore_names.count(func->name_)) continue;
      analyzer.VisitStmt(func->body_);
    }
    auto results = analyzer.results();

    // Collect all standalone call sites (preserving body_index references per
    // orchestration function).
    std::vector<FunctionBodyIndex> body_indices;
    body_indices.reserve(program->functions_.size());
    std::unordered_map<std::string, std::vector<StandaloneCallSite>> standalone_sites;
    for (const auto& [gvar, func] : program->functions_) {
      if (incore_names.count(func->name_)) continue;
      body_indices.emplace_back();
      auto& body_index = body_indices.back();
      body_index.VisitStmt(func->body_);
      StandaloneCallCollector collector(incore_names, body_index, standalone_sites);
      collector.VisitStmt(func->body_);
    }

    // For each callee with standalone call sites, only record a merge if
    // EVERY standalone call site satisfies the pairing's safety preconditions.
    // Caller-dependent safety cannot be cached per-callee otherwise: the
    // rewrite applies globally to every call of that function.
    for (const auto& [fname, sites] : standalone_sites) {
      if (results.count(fname)) continue;  // LoopAnalyzer already handled
      auto pair_it = in_out_pairings.find(fname);
      if (pair_it == in_out_pairings.end() || pair_it->second.empty()) continue;

      AnalysisResult partial;
      partial.func_name = fname;
      std::unordered_set<size_t> seen_out;
      std::unordered_set<size_t> seen_in;

      for (const auto& [in_idx, out_idx] : pair_it->second) {
        bool all_safe = true;
        for (const auto& site : sites) {
          if (!IsPairingSafeAtCallSite(site, in_idx, out_idx)) {
            all_safe = false;
            break;
          }
        }
        if (!all_safe) continue;
        if (!seen_out.insert(out_idx).second) continue;
        if (!seen_in.insert(in_idx).second) continue;
        partial.merges.push_back({out_idx, in_idx});
      }

      if (!partial.merges.empty()) results[fname] = std::move(partial);
    }
    return results;
  }

  // -- Pre-scan: IRVisitor that identifies dead tensor.create vars -----------

  class DeadCreateScanner : public IRVisitor {
   public:
    explicit DeadCreateScanner(const std::unordered_map<std::string, AnalysisResult>& reuse_results)
        : reuse_results_(reuse_results) {}

    const std::unordered_set<const Var*>& dead_creates() const { return dead_creates_; }

   protected:
    void VisitStmt_(const AssignStmtPtr& op) override {
      auto call = As<Call>(op->value_);
      if (!call) return;
      auto fname = GetCallFuncName(call);
      auto reuse_it = reuse_results_.find(fname);
      if (reuse_it == reuse_results_.end()) return;
      for (const auto& merge : reuse_it->second.merges) {
        if (merge.out_param_index < call->args_.size()) {
          if (auto create_var = As<Var>(call->args_[merge.out_param_index])) {
            dead_creates_.insert(create_var.get());
          }
        }
      }
    }

   private:
    const std::unordered_map<std::string, AnalysisResult>& reuse_results_;
    std::unordered_set<const Var*> dead_creates_;
  };

  // -- Mutation: IRMutator that substitutes Var references --------------------

  class VarSubstitutionMutator : public IRMutator {
   public:
    void AddSubstitution(const Var* old_ptr, const VarPtr& new_var) { subs_[old_ptr] = new_var; }

   protected:
    ExprPtr VisitExpr_(const VarPtr& op) override {
      auto it = subs_.find(op.get());
      if (it != subs_.end()) return it->second;
      return op;
    }

   private:
    std::unordered_map<const Var*, VarPtr> subs_;
  };

  /// Rewrite an InCore function to merge Out params into In params.
  FunctionPtr RewriteIncore(const FunctionPtr& func, const std::vector<OutToInMerge>& merges) {
    std::unordered_set<size_t> out_indices_to_remove;
    VarSubstitutionMutator mutator;
    std::unordered_set<size_t> in_indices_to_promote;

    for (const auto& merge : merges) {
      out_indices_to_remove.insert(merge.out_param_index);
      in_indices_to_promote.insert(merge.in_param_index);
      mutator.AddSubstitution(func->params_[merge.out_param_index].get(),
                              func->params_[merge.in_param_index]);
    }

    std::vector<VarPtr> new_params;
    std::vector<ParamDirection> new_directions;
    for (size_t i = 0; i < func->params_.size(); ++i) {
      if (out_indices_to_remove.count(i)) continue;
      new_params.push_back(func->params_[i]);
      if (in_indices_to_promote.count(i)) {
        new_directions.push_back(ParamDirection::InOut);
      } else {
        new_directions.push_back(i < func->param_directions_.size() ? func->param_directions_[i]
                                                                    : ParamDirection::In);
      }
    }

    auto new_body = mutator.VisitStmt(func->body_);

    return std::make_shared<Function>(func->name_, new_params, new_directions, func->return_types_, new_body,
                                      func->span_, func->func_type_, func->level_, func->role_, func->attrs_);
  }

  // -- Mutation: IRMutator that rewrites orch call sites ---------------------

  class CallSiteRewriter : public IRMutator {
   public:
    CallSiteRewriter(const std::unordered_map<std::string, AnalysisResult>& reuse_results,
                     const std::unordered_map<std::string, FunctionPtr>& rewritten_funcs,
                     const std::unordered_set<const Var*>& dead_creates)
        : reuse_results_(reuse_results), rewritten_funcs_(rewritten_funcs), dead_creates_(dead_creates) {}

   protected:
    StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
      auto call = As<Call>(op->value_);
      if (!call) return IRMutator::VisitStmt_(op);

      // Remove dead tensor.create
      if (!std::dynamic_pointer_cast<const GlobalVar>(call->op_) && call->op_->name_ == "tensor.create") {
        if (dead_creates_.count(op->var_.get())) {
          return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, op->span_);
        }
        return IRMutator::VisitStmt_(op);
      }

      // Rewrite calls to rewritten InCore functions
      auto fname = GetCallFuncName(call);
      auto reuse_it = reuse_results_.find(fname);
      if (reuse_it == reuse_results_.end()) return IRMutator::VisitStmt_(op);

      auto func_it = rewritten_funcs_.find(fname);
      if (func_it == rewritten_funcs_.end()) return IRMutator::VisitStmt_(op);

      const auto& merges = reuse_it->second.merges;
      const auto& new_func = func_it->second;

      std::unordered_set<size_t> remove_indices;
      for (const auto& merge : merges) {
        remove_indices.insert(merge.out_param_index);
      }

      std::vector<ExprPtr> new_args;
      for (size_t i = 0; i < call->args_.size(); ++i) {
        if (remove_indices.count(i)) continue;
        new_args.push_back(VisitExpr(call->args_[i]));
      }

      TypePtr new_return_type;
      if (new_func->return_types_.empty()) {
        new_return_type = nullptr;
      } else if (new_func->return_types_.size() == 1) {
        new_return_type = new_func->return_types_[0];
      } else {
        new_return_type = std::make_shared<TupleType>(new_func->return_types_);
      }

      // Preserve attrs_ (e.g. kAttrDumpVars) through the arg-subset rewrite —
      // mirrors the base IRMutator. Only a merged Out param is removed; any
      // dump/dep Var the tag references is a surviving In arg, so a verbatim
      // attr copy stays valid. Fall back to UnknownType for a void-return callee
      // so the rewritten Call's type_ matches the prior 4-arg ctor path.
      auto new_call =
          std::make_shared<Call>(call->op_, new_args, call->kwargs_, call->attrs_,
                                 new_return_type ? new_return_type : GetUnknownType(), call->span_);

      auto new_var = std::make_shared<Var>(op->var_->name_hint_, new_return_type, op->var_->span_);
      var_remap_[op->var_.get()] = new_var;
      auto result = MutableCopy(op);
      result->var_ = new_var;
      result->value_ = new_call;
      return result;
    }

    ExprPtr VisitExpr_(const VarPtr& op) override {
      auto it = var_remap_.find(op.get());
      if (it != var_remap_.end()) return it->second;
      return op;
    }

   private:
    const std::unordered_map<std::string, AnalysisResult>& reuse_results_;
    const std::unordered_map<std::string, FunctionPtr>& rewritten_funcs_;
    std::unordered_set<const Var*> dead_creates_;
    std::unordered_map<const Var*, VarPtr> var_remap_;
  };
};

// ============================================================================
// Pattern 2: AssembleParentStridesOptimizer
//
// Cross-function analysis: scans orchestration for
//   tensor.assemble(parent, incore_result, offset)
// where incore_result comes from an InCore call. Records the parent
// tensor's shape. Then updates the InCore function's Out param
// TensorType to carry parent-derived strides via TensorView.
// ============================================================================

class AssembleParentStridesOptimizer {
 public:
  ProgramPtr Run(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names) {
    auto parent_shapes = Analyze(program, incore_names);
    if (parent_shapes.empty()) return program;

    std::vector<FunctionPtr> new_functions;
    for (const auto& [gvar, func] : program->functions_) {
      new_functions.push_back(func);
    }

    Apply(new_functions, incore_names, parent_shapes);

    return std::make_shared<Program>(new_functions, program->name_, program->span_);
  }

 private:
  using ParentShapeMap = std::unordered_map<std::string, std::unordered_map<size_t, std::vector<ExprPtr>>>;

  // -- Analysis: IRVisitor that tracks InCore call results and finds assemble patterns --

  class AssembleAnalyzer : public IRVisitor {
   public:
    AssembleAnalyzer(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names)
        : program_(program), incore_names_(incore_names) {}

    const ParentShapeMap& result() const { return result_; }

   protected:
    void VisitStmt_(const AssignStmtPtr& op) override {
      auto call = As<Call>(op->value_);
      if (!call) {
        // Check for TupleGetItem extracting from an InCore call result
        auto tgi = As<TupleGetItemExpr>(op->value_);
        if (tgi) {
          auto src_var = AsVarLike(tgi->tuple_);
          if (src_var) {
            auto it = var_to_incore_return_.find(src_var.get());
            if (it != var_to_incore_return_.end()) {
              var_to_incore_return_[op->var_.get()] = {it->second.func_name,
                                                       static_cast<size_t>(tgi->index_)};
            }
          }
        }
        return;
      }

      // Check if this is an InCore call
      auto fname = GetCallFuncName(call);
      if (incore_names_.count(fname)) {
        auto incore_func = FindFunction(program_, fname);
        if (incore_func && incore_func->return_types_.size() == 1) {
          var_to_incore_return_[op->var_.get()] = {fname, 0};
        } else if (incore_func && incore_func->return_types_.size() > 1) {
          var_to_incore_return_[op->var_.get()] = {fname, SIZE_MAX};
        }
        return;
      }

      // Check if this is a tensor.assemble(parent, source, offset)
      if (!std::dynamic_pointer_cast<const GlobalVar>(call->op_) && call->op_->name_ == "tensor.assemble" &&
          call->args_.size() == 3) {
        auto parent_var = AsVarLike(call->args_[0]);
        auto source_var = AsVarLike(call->args_[1]);
        if (!parent_var || !source_var) return;

        auto src_it = var_to_incore_return_.find(source_var.get());
        if (src_it == var_to_incore_return_.end()) return;
        if (src_it->second.return_index == SIZE_MAX) return;

        auto parent_tensor_type = As<TensorType>(parent_var->GetType());
        if (!parent_tensor_type) return;

        result_[src_it->second.func_name][src_it->second.return_index] = parent_tensor_type->shape_;
      }
    }

   private:
    struct IncoreReturnInfo {
      std::string func_name;
      size_t return_index;
    };

    const ProgramPtr& program_;
    const std::unordered_set<std::string>& incore_names_;
    std::unordered_map<const Var*, IncoreReturnInfo> var_to_incore_return_;
    ParentShapeMap result_;
  };

  /// Analyze orchestration functions for tensor.assemble patterns.
  ParentShapeMap Analyze(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names) {
    AssembleAnalyzer analyzer(program, incore_names);
    for (const auto& [gvar, func] : program->functions_) {
      if (incore_names.count(func->name_)) continue;
      analyzer.VisitStmt(func->body_);
    }
    return analyzer.result();
  }

  static std::vector<ExprPtr> ComputeStrides(const std::vector<ExprPtr>& shape) {
    return ComputeRowMajorStrides(shape);
  }

  // -- Mutation: IRMutator that propagates updated param types through tile.store --

  class ParamStrideUpdateMutator : public IRMutator {
   public:
    void AddSubstitution(const Var* old_ptr, const VarPtr& new_var) {
      subs_[old_ptr] = new_var;
      new_param_ptrs_.insert(new_var.get());
    }

   protected:
    ExprPtr VisitExpr_(const VarPtr& op) override {
      auto it = subs_.find(op.get());
      if (it != subs_.end()) return it->second;
      auto remap_it = var_remap_.find(op.get());
      if (remap_it != var_remap_.end()) return remap_it->second;
      return op;
    }

    StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
      auto visited = IRMutator::VisitStmt_(op);
      auto assign = As<AssignStmt>(visited);
      if (!assign) return visited;

      auto call = As<Call>(assign->value_);
      if (!call || call->op_->name_ != "tile.store" || call->args_.size() < 3) return assign;

      auto out_var = AsVarLike(call->args_[2]);
      if (!out_var || !new_param_ptrs_.count(out_var.get())) return assign;

      auto out_type = out_var->GetType();
      auto new_call = std::make_shared<Call>(call->op_, call->args_, call->kwargs_, out_type, call->span_);
      auto new_var = std::make_shared<Var>(assign->var_->name_hint_, out_type, assign->var_->span_);
      var_remap_[op->var_.get()] = new_var;
      auto result = MutableCopy(assign);
      result->var_ = new_var;
      result->value_ = new_call;
      return result;
    }

   private:
    std::unordered_map<const Var*, VarPtr> subs_;
    std::unordered_set<const Var*> new_param_ptrs_;
    std::unordered_map<const Var*, VarPtr> var_remap_;
  };

  /// Apply assemble parent strides to InCore functions.
  void Apply(std::vector<FunctionPtr>& functions, const std::unordered_set<std::string>& incore_names,
             const ParentShapeMap& parent_shapes) {
    for (auto& func : functions) {
      if (!incore_names.count(func->name_)) continue;

      auto ps_it = parent_shapes.find(func->name_);
      if (ps_it == parent_shapes.end()) continue;
      const auto& return_idx_to_shape = ps_it->second;

      auto out_mappings = BuildOutParamReturnMappings(func);
      if (out_mappings.empty()) continue;

      bool changed = false;
      std::vector<VarPtr> new_params = func->params_;

      for (const auto& opm : out_mappings) {
        auto shape_it = return_idx_to_shape.find(opm.return_index);
        if (shape_it == return_idx_to_shape.end()) continue;

        auto full_strides = ComputeStrides(shape_it->second);
        if (full_strides.empty()) continue;

        auto tensor_type = As<TensorType>(func->params_[opm.param_index]->GetType());
        if (!tensor_type) continue;

        // Extract trailing strides matching the output tensor's rank.
        // For a 3D parent [B, M, N] with strides [M*N, N, 1] and a 2D output [M', N'],
        // we need the last 2 strides: [N, 1].
        size_t out_rank = tensor_type->shape_.size();
        if (out_rank > full_strides.size()) continue;
        std::vector<ExprPtr> strides(full_strides.end() - static_cast<std::ptrdiff_t>(out_rank),
                                     full_strides.end());

        TensorView view(std::move(strides), TensorLayout::ND);
        auto new_type = std::make_shared<TensorType>(tensor_type->shape_, tensor_type->dtype_,
                                                     tensor_type->memref_, std::move(view));
        auto new_param = std::make_shared<Var>(func->params_[opm.param_index]->name_hint_, new_type,
                                               func->params_[opm.param_index]->span_);

        changed = true;
        new_params[opm.param_index] = new_param;
      }

      if (!changed) continue;

      ParamStrideUpdateMutator mutator;
      for (size_t i = 0; i < func->params_.size(); ++i) {
        if (new_params[i].get() != func->params_[i].get()) {
          mutator.AddSubstitution(func->params_[i].get(), new_params[i]);
        }
      }
      auto new_body = mutator.VisitStmt(func->body_);

      func = std::make_shared<Function>(func->name_, new_params, func->param_directions_, func->return_types_,
                                        new_body, func->span_, func->func_type_, func->level_, func->role_,
                                        func->attrs_);
    }
  }
};

// ============================================================================
// Pattern 3: AssembleLoopRewriter
//
// InCore-local optimization: when an InCore function body has a ForStmt
// that does tile.assemble accumulation yielding back to iter-arg, and the
// ForStmt result feeds the final tile.store -> return, rewrite the loop
// to use tile.store instead of tile.assemble, with the iter-arg initialized
// from the Out param.
// ============================================================================

class AssembleLoopRewriter {
 public:
  ProgramPtr Run(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names) {
    std::vector<FunctionPtr> new_functions;
    bool changed = false;

    for (const auto& [gvar, func] : program->functions_) {
      if (!incore_names.count(func->name_)) {
        new_functions.push_back(func);
        continue;
      }
      bool has_out = false;
      for (const auto& dir : func->param_directions_) {
        if (dir == ParamDirection::Out) {
          has_out = true;
          break;
        }
      }
      if (!has_out) {
        new_functions.push_back(func);
        continue;
      }
      auto rewritten = RewriteFunction(func);
      if (rewritten.get() != func.get()) changed = true;
      new_functions.push_back(rewritten);
    }

    if (!changed) return program;
    return std::make_shared<Program>(new_functions, program->name_, program->span_);
  }

 private:
  /// Check if a statement subtree uses a given Var (by raw pointer).
  /// Uses VarDefUseCollector which handles all statement/expression types.
  static bool StmtUsesVar(const StmtPtr& stmt, const Var* var) {
    if (!stmt || !var) return false;
    var_collectors::VarDefUseCollector collector;
    collector.VisitStmt(stmt);
    return collector.var_uses.count(var) > 0;
  }

  // -- Pre-scan: IRVisitor that collects var definitions and return statement --

  class BodyScanner : public IRVisitor {
   public:
    const std::unordered_map<const Var*, AssignStmtPtr>& var_def() const { return var_def_; }
    const ReturnStmtPtr& return_stmt() const { return return_stmt_; }

   protected:
    void VisitStmt_(const AssignStmtPtr& op) override { var_def_[op->var_.get()] = op; }
    void VisitStmt_(const ReturnStmtPtr& op) override { return_stmt_ = op; }

   private:
    std::unordered_map<const Var*, AssignStmtPtr> var_def_;
    ReturnStmtPtr return_stmt_;
  };

  // -- Mutation: IRMutator that rewrites matching ForStmt patterns -----------

  class LoopRewriteMutator : public IRMutator {
   public:
    struct StoreReturnInfo {
      const Var* store_var;
      const Var* out_param;
      size_t return_index;
    };

    LoopRewriteMutator(const std::unordered_map<const Var*, size_t>& out_var_to_param_idx,
                       const std::unordered_map<const Var*, const StoreReturnInfo*>& for_return_to_store,
                       const std::unordered_set<const Var*>& dead_create_vars,
                       const std::unordered_set<const Var*>& dead_store_vars,
                       const std::unordered_map<const Var*, VarPtr>& return_var_remap,
                       const FunctionPtr& func)
        : out_var_to_param_idx_(out_var_to_param_idx),
          for_return_to_store_(for_return_to_store),
          dead_create_vars_(dead_create_vars),
          dead_store_vars_(dead_store_vars),
          return_var_remap_(return_var_remap),
          func_(func) {}

   protected:
    StmtPtr VisitStmt_(const ForStmtPtr& op) override {
      if (op->iter_args_.size() != 1 || op->return_vars_.size() != 1) {
        return IRMutator::VisitStmt_(op);
      }

      auto fret_it = for_return_to_store_.find(op->return_vars_[0].get());
      if (fret_it == for_return_to_store_.end()) {
        return IRMutator::VisitStmt_(op);
      }
      const auto& store_info = *fret_it->second;

      auto loop_body_stmts = FlattenToStmts(op->body_);
      auto yield = transform_utils::FindYieldStmt(op->body_);
      if (!yield || yield->value_.size() != 1) {
        return IRMutator::VisitStmt_(op);
      }

      const IterArg* iter_arg = op->iter_args_[0].get();

      // Find the tile.assemble call
      AssignStmtPtr assemble_assign;
      for (const auto& body_stmt : loop_body_stmts) {
        auto assign = As<AssignStmt>(body_stmt);
        if (!assign) continue;
        auto call = As<Call>(assign->value_);
        if (!call || call->op_->name_ != "tile.assemble") continue;
        if (call->args_.size() < 3) continue;
        const Var* arg0_raw = nullptr;
        if (auto v = As<Var>(call->args_[0])) arg0_raw = v.get();
        if (auto ia = As<IterArg>(call->args_[0])) arg0_raw = ia.get();
        if (arg0_raw != iter_arg) continue;
        assemble_assign = assign;
        break;
      }

      if (!assemble_assign) return IRMutator::VisitStmt_(op);

      // --- Rewrite: tile.assemble -> tile.store ---

      auto out_param_var = func_->params_[out_var_to_param_idx_.at(store_info.out_param)];
      auto out_tensor_type = As<TensorType>(out_param_var->GetType());
      INTERNAL_CHECK_SPAN(out_tensor_type, out_param_var->span_)
          << "Internal error: Out param should be TensorType";

      auto new_iter_arg = std::make_shared<IterArg>(op->iter_args_[0]->name_hint_, out_tensor_type,
                                                    out_param_var, op->iter_args_[0]->span_);

      auto assemble_call = As<Call>(assemble_assign->value_);
      auto& op_registry = OpRegistry::GetInstance();
      auto store_call =
          op_registry.Create("tile.store", {assemble_call->args_[1], assemble_call->args_[2], new_iter_arg},
                             assemble_assign->value_->span_);
      auto store_result_var = std::make_shared<Var>(assemble_assign->var_->name_hint_, store_call->GetType(),
                                                    assemble_assign->span_);

      std::vector<StmtPtr> new_loop_stmts;
      for (const auto& body_stmt : loop_body_stmts) {
        if (body_stmt.get() == assemble_assign.get()) {
          auto store_assign = MutableCopy(assemble_assign);
          store_assign->var_ = store_result_var;
          store_assign->value_ = store_call;
          new_loop_stmts.push_back(std::move(store_assign));
        } else if (auto y = As<YieldStmt>(body_stmt)) {
          auto new_yield = MutableCopy(y);
          new_yield->value_ = std::vector<ExprPtr>{store_result_var};
          new_loop_stmts.push_back(std::move(new_yield));
        } else {
          new_loop_stmts.push_back(body_stmt);
        }
      }

      auto new_loop_body = SeqStmts::Flatten(std::move(new_loop_stmts), op->body_->span_);
      auto new_return_var = return_var_remap_.at(store_info.store_var);

      auto result = MutableCopy(op);
      result->iter_args_ = std::vector<IterArgPtr>{new_iter_arg};
      result->body_ = new_loop_body;
      result->return_vars_ = std::vector<VarPtr>{new_return_var};
      return result;
    }

    StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
      if (dead_create_vars_.count(op->var_.get()) || dead_store_vars_.count(op->var_.get())) {
        return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, op->span_);
      }
      return IRMutator::VisitStmt_(op);
    }

    StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
      auto visited = IRMutator::VisitStmt_(op);
      auto seq = As<SeqStmts>(visited);
      if (!seq) return visited;
      // Filter out empty SeqStmts children (from deleted statements)
      std::vector<StmtPtr> filtered;
      for (const auto& s : seq->stmts_) {
        if (auto child_seq = As<SeqStmts>(s)) {
          if (child_seq->stmts_.empty()) continue;
        }
        filtered.push_back(s);
      }
      if (filtered.size() == seq->stmts_.size()) return seq;
      return SeqStmts::Flatten(std::move(filtered), seq->span_);
    }

    StmtPtr VisitStmt_(const ReturnStmtPtr& op) override {
      if (return_var_remap_.empty()) return op;
      std::vector<ExprPtr> new_ret_values;
      bool remapped = false;
      for (const auto& v : op->value_) {
        auto var = As<Var>(v);
        if (var) {
          auto remap_it = return_var_remap_.find(var.get());
          if (remap_it != return_var_remap_.end()) {
            new_ret_values.push_back(remap_it->second);
            remapped = true;
            continue;
          }
        }
        new_ret_values.push_back(v);
      }
      if (!remapped) return op;
      auto result = MutableCopy(op);
      result->value_ = std::move(new_ret_values);
      return result;
    }

   private:
    const std::unordered_map<const Var*, size_t>& out_var_to_param_idx_;
    const std::unordered_map<const Var*, const StoreReturnInfo*>& for_return_to_store_;
    const std::unordered_set<const Var*>& dead_create_vars_;
    const std::unordered_set<const Var*>& dead_store_vars_;
    const std::unordered_map<const Var*, VarPtr>& return_var_remap_;
    const FunctionPtr& func_;
  };

  /// Rewrite assemble-loop pattern in an InCore function.
  FunctionPtr RewriteFunction(const FunctionPtr& func) {
    // Pre-scan: collect var definitions and return statement
    BodyScanner scanner;
    scanner.VisitStmt(func->body_);
    const auto& var_def = scanner.var_def();
    const auto& return_stmt = scanner.return_stmt();
    if (!return_stmt) return func;

    // Identify Out params
    std::unordered_map<const Var*, size_t> out_var_to_param_idx;
    for (size_t i = 0; i < func->params_.size(); ++i) {
      if (i < func->param_directions_.size() && func->param_directions_[i] == ParamDirection::Out) {
        out_var_to_param_idx[func->params_[i].get()] = i;
      }
    }
    if (out_var_to_param_idx.empty()) return func;

    // Map return values -> tile.store -> Out params
    using StoreReturnInfo = LoopRewriteMutator::StoreReturnInfo;
    std::vector<StoreReturnInfo> store_returns;

    for (size_t ret_i = 0; ret_i < return_stmt->value_.size(); ++ret_i) {
      auto ret_var = As<Var>(return_stmt->value_[ret_i]);
      if (!ret_var) continue;
      auto def_it = var_def.find(ret_var.get());
      if (def_it == var_def.end()) continue;
      auto call = As<Call>(def_it->second->value_);
      if (!call || call->op_->name_ != "tile.store") continue;
      if (call->args_.size() < 3) continue;
      auto out_tensor = As<Var>(call->args_[2]);
      if (!out_tensor || !out_var_to_param_idx.count(out_tensor.get())) continue;
      store_returns.push_back({ret_var.get(), out_tensor.get(), ret_i});
    }
    if (store_returns.empty()) return func;

    // Map ForStmt return_var -> which store_return it feeds
    std::unordered_map<const Var*, const StoreReturnInfo*> for_return_to_store;
    for (const auto& sr : store_returns) {
      auto def_it = var_def.find(sr.store_var);
      if (def_it == var_def.end()) continue;
      auto call = As<Call>(def_it->second->value_);
      if (!call || call->op_->name_ != "tile.store") continue;
      auto tile_data_var = As<Var>(call->args_[0]);
      if (!tile_data_var) continue;
      for_return_to_store[tile_data_var.get()] = &sr;
    }

    // Pre-compute dead sets by scanning ForStmts for pattern matches.
    // This must happen before the IRMutator pass because dead tile.create
    // statements may appear before the ForStmt they correspond to.
    std::unordered_set<const Var*> dead_create_vars;
    std::unordered_set<const Var*> dead_store_vars;
    std::unordered_map<const Var*, VarPtr> return_var_remap;

    class ForStmtMatchScanner : public IRVisitor {
     public:
      ForStmtMatchScanner(const std::unordered_map<const Var*, const StoreReturnInfo*>& for_return_to_store,
                          const std::unordered_map<const Var*, size_t>& out_var_to_param_idx,
                          const FunctionPtr& func, std::unordered_set<const Var*>& dead_create_vars,
                          std::unordered_set<const Var*>& dead_store_vars,
                          std::unordered_map<const Var*, VarPtr>& return_var_remap)
          : for_return_to_store_(for_return_to_store),
            out_var_to_param_idx_(out_var_to_param_idx),
            func_(func),
            dead_create_vars_(dead_create_vars),
            dead_store_vars_(dead_store_vars),
            return_var_remap_(return_var_remap) {}

      [[nodiscard]] bool matched() const { return matched_; }

     protected:
      void VisitStmt_(const ForStmtPtr& op) override {
        IRVisitor::VisitStmt_(op);
        if (op->iter_args_.size() != 1 || op->return_vars_.size() != 1) return;

        auto fret_it = for_return_to_store_.find(op->return_vars_[0].get());
        if (fret_it == for_return_to_store_.end()) return;
        const auto& store_info = *fret_it->second;

        auto loop_body_stmts = FlattenToStmts(op->body_);
        auto yield = transform_utils::FindYieldStmt(op->body_);
        if (!yield || yield->value_.size() != 1) return;

        const IterArg* iter_arg = op->iter_args_[0].get();

        AssignStmtPtr assemble_assign;
        for (const auto& body_stmt : loop_body_stmts) {
          auto assign = As<AssignStmt>(body_stmt);
          if (!assign) continue;
          auto call = As<Call>(assign->value_);
          if (!call || call->op_->name_ != "tile.assemble") continue;
          if (call->args_.size() < 3) continue;
          const Var* arg0_raw = nullptr;
          if (auto v = As<Var>(call->args_[0])) arg0_raw = v.get();
          if (auto ia = As<IterArg>(call->args_[0])) arg0_raw = ia.get();
          if (arg0_raw != iter_arg) continue;
          assemble_assign = assign;
          break;
        }
        if (!assemble_assign) return;

        auto yield_var = As<Var>(yield->value_[0]);
        if (!yield_var || yield_var.get() != assemble_assign->var_.get()) return;

        bool iter_arg_used_elsewhere = false;
        for (const auto& body_stmt : loop_body_stmts) {
          if (body_stmt.get() == assemble_assign.get()) continue;
          if (As<YieldStmt>(body_stmt)) continue;
          if (StmtUsesVar(body_stmt, iter_arg)) {
            iter_arg_used_elsewhere = true;
            break;
          }
        }
        if (iter_arg_used_elsewhere) return;

        // Pattern matched — record dead sets
        matched_ = true;
        auto init_var = As<Var>(op->iter_args_[0]->initValue_);
        if (init_var) dead_create_vars_.insert(init_var.get());
        dead_store_vars_.insert(store_info.store_var);

        auto out_param_var = func_->params_[out_var_to_param_idx_.at(store_info.out_param)];
        auto out_tensor_type = As<TensorType>(out_param_var->GetType());
        INTERNAL_CHECK_SPAN(out_tensor_type, out_param_var->span_)
            << "Internal error: Out param should be TensorType";
        auto new_return_var = std::make_shared<Var>(op->return_vars_[0]->name_hint_, out_tensor_type,
                                                    op->return_vars_[0]->span_);
        return_var_remap_[store_info.store_var] = new_return_var;
      }

     private:
      const std::unordered_map<const Var*, const StoreReturnInfo*>& for_return_to_store_;
      const std::unordered_map<const Var*, size_t>& out_var_to_param_idx_;
      const FunctionPtr& func_;
      std::unordered_set<const Var*>& dead_create_vars_;
      std::unordered_set<const Var*>& dead_store_vars_;
      std::unordered_map<const Var*, VarPtr>& return_var_remap_;
      bool matched_ = false;
    };

    ForStmtMatchScanner match_scanner(for_return_to_store, out_var_to_param_idx, func, dead_create_vars,
                                      dead_store_vars, return_var_remap);
    match_scanner.VisitStmt(func->body_);
    if (!match_scanner.matched()) return func;

    // Apply the IRMutator using pre-computed dead sets
    LoopRewriteMutator mutator(out_var_to_param_idx, for_return_to_store, dead_create_vars, dead_store_vars,
                               return_var_remap, func);
    auto new_body = mutator.VisitStmt(func->body_);

    return std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                      func->return_types_, new_body, func->span_, func->func_type_,
                                      func->level_, func->role_, func->attrs_);
  }
};

// ============================================================================
// Pattern 4: SliceInputStridesOptimizer
//
// Cross-function analysis: scans orchestration for
//   tensor.slice(parent, size, offset)
// where the slice result is passed as an In argument to an InCore call.
// Records the parent tensor's shape. Then updates the InCore function's
// In param TensorType to carry parent-derived strides via TensorView.
// ============================================================================

class SliceInputStridesOptimizer {
 public:
  ProgramPtr Run(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names) {
    auto input_shapes = Analyze(program, incore_names);
    if (input_shapes.empty()) return program;

    std::vector<FunctionPtr> new_functions;
    for (const auto& [gvar, func] : program->functions_) {
      new_functions.push_back(func);
    }

    Apply(new_functions, incore_names, input_shapes);

    return std::make_shared<Program>(new_functions, program->name_, program->span_);
  }

 private:
  // func_name -> { param_index -> parent_shape }
  using InputShapeMap = std::unordered_map<std::string, std::unordered_map<size_t, std::vector<ExprPtr>>>;

  static bool ShapesMatch(const std::vector<ExprPtr>& a, const std::vector<ExprPtr>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
      auto ca = As<ConstInt>(a[i]);
      auto cb = As<ConstInt>(b[i]);
      if (!ca || !cb || ca->value_ != cb->value_) return false;
    }
    return true;
  }

  class SliceAnalyzer : public IRVisitor {
   public:
    SliceAnalyzer(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names)
        : program_(program), incore_names_(incore_names) {}

    const InputShapeMap& result() const { return result_; }

   protected:
    void VisitStmt_(const AssignStmtPtr& op) override {
      auto call = As<Call>(op->value_);
      if (!call) return;

      // Track tensor.slice(parent, size, offset) results
      if (!std::dynamic_pointer_cast<const GlobalVar>(call->op_) && call->op_->name_ == "tensor.slice" &&
          call->args_.size() >= 3) {
        auto parent_var = AsVarLike(call->args_[0]);
        if (parent_var) {
          auto parent_tensor_type = As<TensorType>(parent_var->GetType());
          if (parent_tensor_type) {
            var_to_parent_shape_[op->var_.get()] = parent_tensor_type->shape_;
          }
        }
        return;
      }

      // Check InCore calls: map sliced In arguments to parent shapes
      auto fname = GetCallFuncName(call);
      if (!incore_names_.count(fname)) return;

      auto incore_func = FindFunction(program_, fname);
      if (!incore_func) return;

      for (size_t i = 0; i < call->args_.size() && i < incore_func->param_directions_.size(); ++i) {
        if (incore_func->param_directions_[i] != ParamDirection::In) continue;

        auto arg_var = AsVarLike(call->args_[i]);
        if (!arg_var) continue;

        auto it = var_to_parent_shape_.find(arg_var.get());
        if (it == var_to_parent_shape_.end()) continue;

        auto conflict_key = fname + ":" + std::to_string(i);
        if (conflicted_.count(conflict_key)) continue;

        auto& entry = result_[fname][i];
        if (entry.empty()) {
          entry = it->second;
        } else if (!ShapesMatch(entry, it->second)) {
          conflicted_.insert(conflict_key);
          entry.clear();
        }
      }
    }

   private:
    const ProgramPtr& program_;
    const std::unordered_set<std::string>& incore_names_;
    std::unordered_map<const Var*, std::vector<ExprPtr>> var_to_parent_shape_;
    std::unordered_set<std::string> conflicted_;
    InputShapeMap result_;
  };

  InputShapeMap Analyze(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names) {
    SliceAnalyzer analyzer(program, incore_names);
    for (const auto& [gvar, func] : program->functions_) {
      if (incore_names.count(func->name_)) continue;
      analyzer.VisitStmt(func->body_);
    }
    return analyzer.result();
  }

  class InParamSubstitutionMutator : public IRMutator {
   public:
    void AddSubstitution(const Var* old_ptr, const VarPtr& new_var) { subs_[old_ptr] = new_var; }

   protected:
    ExprPtr VisitExpr_(const VarPtr& op) override {
      auto it = subs_.find(op.get());
      if (it != subs_.end()) return it->second;
      return op;
    }

   private:
    std::unordered_map<const Var*, VarPtr> subs_;
  };

  void Apply(std::vector<FunctionPtr>& functions, const std::unordered_set<std::string>& incore_names,
             const InputShapeMap& input_shapes) {
    for (auto& func : functions) {
      if (!incore_names.count(func->name_)) continue;

      auto is_it = input_shapes.find(func->name_);
      if (is_it == input_shapes.end()) continue;
      const auto& param_idx_to_shape = is_it->second;

      bool changed = false;
      std::vector<VarPtr> new_params = func->params_;

      for (const auto& [param_idx, parent_shape] : param_idx_to_shape) {
        if (parent_shape.empty()) continue;  // conflicted or not from slice
        if (param_idx >= func->params_.size()) continue;

        auto full_strides = ComputeRowMajorStrides(parent_shape);
        if (full_strides.empty()) continue;

        auto tensor_type = As<TensorType>(func->params_[param_idx]->GetType());
        if (!tensor_type) continue;

        // Skip params that already have explicit strides
        if (tensor_type->tensor_view_.has_value() && !tensor_type->tensor_view_->stride.empty()) continue;

        size_t in_rank = tensor_type->shape_.size();
        if (in_rank > full_strides.size()) continue;
        std::vector<ExprPtr> strides(full_strides.end() - static_cast<std::ptrdiff_t>(in_rank),
                                     full_strides.end());

        TensorView view(std::move(strides), TensorLayout::ND);
        auto new_type = std::make_shared<TensorType>(tensor_type->shape_, tensor_type->dtype_,
                                                     tensor_type->memref_, std::move(view));
        auto new_param = std::make_shared<Var>(func->params_[param_idx]->name_hint_, new_type,
                                               func->params_[param_idx]->span_);

        changed = true;
        new_params[param_idx] = new_param;
      }

      if (!changed) continue;

      InParamSubstitutionMutator mutator;
      for (size_t i = 0; i < func->params_.size(); ++i) {
        if (new_params[i].get() != func->params_[i].get()) {
          mutator.AddSubstitution(func->params_[i].get(), new_params[i]);
        }
      }
      auto new_body = mutator.VisitStmt(func->body_);

      func = std::make_shared<Function>(func->name_, new_params, func->param_directions_, func->return_types_,
                                        new_body, func->span_, func->func_type_, func->level_, func->role_,
                                        func->attrs_);
    }
  }
};

// ============================================================================
// Pattern 5: Static output-window planning
//
// Rewrites statically provable local-window writes into explicit
// slice -> windowed callee -> assemble structure at the orchestration callsite.
//
// Supported shapes:
// - FinalStore: single call writes one final local window of an Out param
// - AggregateWindowLoop: an outlined non-builtin callee writes a loop-carried
//   aggregate window into one or more Out params, and every rewritten Out can
//   be proven disjoint across sequential sibling callsites.
//
// Multi-Out policy is all-or-nothing: either every Out param is rewritten, or
// the callee stays baseline.
// ============================================================================

class StaticOutputWindowPlanner {
 public:
  ProgramPtr Run(const ProgramPtr& program) {
    auto analyses = Analyze(program);
    if (analyses.empty()) return program;

    std::unordered_map<std::string, FunctionPtr> cloned_funcs;
    for (const auto& [func_name, analysis] : analyses) {
      auto callee = program->GetFunction(func_name);
      if (!callee) continue;
      auto cloned = RewriteCallee(program, callee, analysis);
      if (!cloned) continue;
      cloned_funcs.emplace(func_name, cloned);
    }
    if (cloned_funcs.empty()) return program;

    std::vector<FunctionPtr> new_functions;
    new_functions.reserve(program->functions_.size() + cloned_funcs.size());
    for (const auto& [gvar, func] : program->functions_) {
      new_functions.push_back(func);
      auto clone_it = cloned_funcs.find(func->name_);
      if (clone_it != cloned_funcs.end()) {
        new_functions.push_back(clone_it->second);
      }
    }

    bool changed = false;
    for (auto& func : new_functions) {
      if (!func || func->func_type_ != FunctionType::Orchestration) continue;
      std::vector<PlannedWindowGroup> planned_windows;
      std::vector<WindowWriteRecord> writer_records;
      OrchRewriter rewriter(program, analyses, cloned_funcs, &planned_windows, &writer_records);
      auto new_body = rewriter.VisitStmt(func->body_);
      if (!planned_windows.empty()) {
        StaticWindowMaterializer materializer(std::move(planned_windows));
        new_body = materializer.VisitStmt(new_body);
      }
      if (!writer_records.empty()) {
        WindowDependencyPlanner planner(program, std::move(writer_records));
        new_body = planner.VisitStmt(new_body);
      }
      if (new_body.get() == func->body_.get()) continue;
      changed = true;
      func = std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                        func->return_types_, new_body, func->span_, func->func_type_,
                                        func->level_, func->role_, func->attrs_);
    }

    if (!changed) return program;
    return std::make_shared<Program>(new_functions, program->comm_groups_, program->name_, program->span_);
  }

 private:
  enum class RewriteKind {
    FinalStore,
    AggregateWindowLoop,
  };

  struct OutputRewriteInfo {
    size_t out_param_index;
    size_t return_index;
    std::vector<ExprPtr> parent_shape;
    std::vector<ExprPtr> window_shape;
    std::vector<ExprPtr> callsite_offsets;
    std::vector<ExprPtr> local_store_offsets;
    size_t iter_arg_index = SIZE_MAX;
  };

  struct CalleeRewriteAnalysis {
    RewriteKind kind = RewriteKind::FinalStore;
    std::vector<OutputRewriteInfo> outputs;
  };

  using AnalysisMap = std::unordered_map<std::string, CalleeRewriteAnalysis>;

  struct PlannedWindowGroup {
    std::string group_id;
    std::string array_name;
    VarPtr parent;
    std::vector<ExprPtr> shape;
    std::vector<ExprPtr> offset;
    std::vector<ForStmtPtr> loop_stack;
    std::vector<VarPtr> emitted_loop_vars;
    std::vector<ExprPtr> emitted_loop_starts;
    std::vector<ExprPtr> emitted_loop_steps;
    std::vector<int64_t> emitted_loop_trips;
  };

  struct RegionDim {
    ExprPtr extent;
    ExprPtr stride;
  };

  struct CanonicalRegion {
    VarPtr root;
    std::vector<ExprPtr> root_shape;
    ExprPtr base_linear;
    std::vector<RegionDim> dims;
  };

  struct WindowWriteRecord {
    VarPtr task_id_var;
    std::string group_id;
    VarPtr parent;
    ForStmtPtr enclosing_loop;
    VarPtr enclosing_loop_var;
    ExprPtr window_index_expr;
    CanonicalRegion write_region;
  };

  struct AffineForm {
    int64_t coeff = 0;
    ExprPtr base;
  };

  struct OrderedLoopOffsets {
    ExprPtr min;
    ExprPtr max;
  };

  static std::optional<AffineForm> ParseAffineInLoop(const ExprPtr& expr, const Var* loop_var) {
    if (!expr) return std::nullopt;
    if (auto ci = As<ConstInt>(expr)) {
      return AffineForm{0, expr};
    }
    if (auto var = AsVarLike(expr)) {
      if (var.get() == loop_var) {
        auto zero = std::make_shared<ConstInt>(0, DataType::INDEX, expr->span_);
        return AffineForm{1, zero};
      }
      return AffineForm{0, expr};
    }
    if (auto add = As<Add>(expr)) {
      auto lhs = ParseAffineInLoop(add->left_, loop_var);
      auto rhs = ParseAffineInLoop(add->right_, loop_var);
      if (!lhs.has_value() || !rhs.has_value()) return std::nullopt;
      return AffineForm{lhs->coeff + rhs->coeff, MakeAdd(lhs->base, rhs->base, expr->span_)};
    }
    if (auto sub = As<Sub>(expr)) {
      auto lhs = ParseAffineInLoop(sub->left_, loop_var);
      auto rhs = ParseAffineInLoop(sub->right_, loop_var);
      if (!lhs.has_value() || !rhs.has_value()) return std::nullopt;
      return AffineForm{lhs->coeff - rhs->coeff, MakeSub(lhs->base, rhs->base, expr->span_)};
    }
    if (auto mul = As<Mul>(expr)) {
      auto lhs_ci = As<ConstInt>(mul->left_);
      auto rhs_ci = As<ConstInt>(mul->right_);
      if (lhs_ci) {
        auto rhs = ParseAffineInLoop(mul->right_, loop_var);
        if (!rhs.has_value()) return std::nullopt;
        return AffineForm{lhs_ci->value_ * rhs->coeff,
                          MakeMul(std::make_shared<ConstInt>(lhs_ci->value_, lhs_ci->dtype(), lhs_ci->span_),
                                  rhs->base, expr->span_)};
      }
      if (rhs_ci) {
        auto lhs = ParseAffineInLoop(mul->left_, loop_var);
        if (!lhs.has_value()) return std::nullopt;
        return AffineForm{
            rhs_ci->value_ * lhs->coeff,
            MakeMul(lhs->base, std::make_shared<ConstInt>(rhs_ci->value_, rhs_ci->dtype(), rhs_ci->span_),
                    expr->span_)};
      }
    }
    return std::nullopt;
  }

  class WindowWriteLocalizer : public IRMutator {
   public:
    WindowWriteLocalizer(const std::unordered_map<const Var*, OutputRewriteInfo>& out_info_by_var,
                         const std::unordered_map<const Var*, ExprPtr>& new_out_vars,
                         const std::unordered_map<const Var*, TypePtr>& new_store_types)
        : out_info_by_var_(out_info_by_var), new_out_vars_(new_out_vars), new_store_types_(new_store_types) {}

   protected:
    ExprPtr VisitExpr_(const VarPtr& op) override {
      auto remap_it = result_var_remap_.find(op.get());
      if (remap_it != result_var_remap_.end()) return remap_it->second;
      auto out_it = new_out_vars_.find(op.get());
      if (out_it != new_out_vars_.end()) return out_it->second;
      return IRMutator::VisitExpr_(op);
    }

    ExprPtr VisitExpr_(const IterArgPtr& op) override {
      auto out_it = new_out_vars_.find(op.get());
      if (out_it != new_out_vars_.end()) return out_it->second;
      return IRMutator::VisitExpr_(op);
    }

    StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
      auto visited_value = VisitExpr(op->value_);
      auto assign = MutableCopy(op);
      assign->value_ = visited_value;
      auto call = As<Call>(assign->value_);
      if (!call) return assign;

      ExprPtr rewritten_target_expr;
      const Var* target_var = nullptr;
      MakeTuplePtr offsets;
      size_t offset_arg_index = SIZE_MAX;
      size_t target_arg_index = SIZE_MAX;

      if (call->op_->name_ == "tile.store" && call->args_.size() >= 3) {
        rewritten_target_expr = call->args_[2];
        auto out_var = AsVarLike(rewritten_target_expr);
        if (!out_var) return assign;
        target_var = out_var.get();
        offsets = As<MakeTuple>(call->args_[1]);
        offset_arg_index = 1;
        target_arg_index = 2;
      } else if (call->op_->name_ == "tensor.assemble" && call->args_.size() >= 3) {
        rewritten_target_expr = call->args_[0];
        auto parent_var = AsVarLike(rewritten_target_expr);
        if (!parent_var) return assign;
        target_var = parent_var.get();
        offsets = As<MakeTuple>(call->args_[2]);
        offset_arg_index = 2;
        target_arg_index = 0;
      } else {
        return assign;
      }

      auto info_it = out_info_by_var_.find(target_var);
      if (info_it == out_info_by_var_.end()) return assign;
      if (!offsets) return assign;

      auto new_offset_tuple =
          std::make_shared<MakeTuple>(info_it->second.local_store_offsets, offsets->span_);
      std::vector<ExprPtr> new_args = call->args_;
      new_args[offset_arg_index] = new_offset_tuple;
      new_args[target_arg_index] = new_out_vars_.at(target_var);
      auto new_type = new_store_types_.at(target_var);
      auto new_call =
          std::make_shared<Call>(call->op_, new_args, call->kwargs_, call->attrs_, new_type, call->span_);

      auto new_result_var = std::make_shared<Var>(assign->var_->name_hint_, new_type, assign->var_->span_);
      result_var_remap_[assign->var_.get()] = new_result_var;
      assign->var_ = new_result_var;
      assign->value_ = new_call;
      return assign;
    }

   private:
    const std::unordered_map<const Var*, OutputRewriteInfo>& out_info_by_var_;
    const std::unordered_map<const Var*, ExprPtr>& new_out_vars_;
    const std::unordered_map<const Var*, TypePtr>& new_store_types_;
    std::unordered_map<const Var*, VarPtr> result_var_remap_;
  };

  class OrchRewriter : public IRMutator {
   public:
    OrchRewriter(ProgramPtr program, const AnalysisMap& analyses,
                 const std::unordered_map<std::string, FunctionPtr>& cloned_funcs,
                 std::vector<PlannedWindowGroup>* planned_windows,
                 std::vector<WindowWriteRecord>* writer_records)
        : program_(std::move(program)),
          analyses_(analyses),
          cloned_funcs_(cloned_funcs),
          planned_windows_(planned_windows),
          writer_records_(writer_records) {}

   protected:
    StmtPtr VisitStmt_(const ForStmtPtr& op) override {
      auto saved_loop_iter_init_subst = loop_iter_init_subst_;
      for (const auto& iter_arg : op->iter_args_) {
        if (iter_arg && iter_arg->initValue_) loop_iter_init_subst_[iter_arg.get()] = iter_arg->initValue_;
      }

      loop_stack_.push_back(op);
      bool is_sequential = op->kind_ != ForKind::Parallel;
      if (is_sequential) {
        sequential_loops_.push_back(op);
        loop_local_allocs_.emplace_back(CollectLoopLocalTensorAllocs(op));
      }
      auto result = IRMutator::VisitStmt_(op);
      if (is_sequential) {
        loop_local_allocs_.pop_back();
        sequential_loops_.pop_back();
      }
      loop_stack_.pop_back();
      loop_iter_init_subst_ = std::move(saved_loop_iter_init_subst);
      return result;
    }

    StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
      ++while_depth_;
      auto result = IRMutator::VisitStmt_(op);
      --while_depth_;
      return result;
    }

    StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
      std::vector<StmtPtr> new_stmts;
      new_stmts.reserve(op->stmts_.size());
      bool changed = false;
      auto saved_scalar_defs = scalar_defs_;
      auto saved_tuple_result_subst = tuple_result_subst_;

      for (const auto& stmt : op->stmts_) {
        auto call_assign = As<AssignStmt>(stmt);
        auto bundle = call_assign ? TryRewriteCall(call_assign) : std::nullopt;
        if (bundle.has_value()) {
          changed = true;
          for (const auto& new_stmt : bundle->stmts) {
            new_stmts.push_back(VisitStmt(new_stmt));
          }
          continue;
        }

        auto visited = VisitStmt(stmt);
        changed = changed || visited.get() != stmt.get();
        new_stmts.push_back(visited);

        auto visited_assign = As<AssignStmt>(visited);
        if (visited_assign && As<ScalarType>(visited_assign->var_->GetType())) {
          scalar_defs_[visited_assign->var_.get()] = visited_assign->value_;
        }
      }

      scalar_defs_ = std::move(saved_scalar_defs);
      tuple_result_subst_ = std::move(saved_tuple_result_subst);
      if (!changed) return op;
      return SeqStmts::Flatten(std::move(new_stmts), op->span_);
    }

   private:
    struct SliceBundle {
      VarPtr slice_var;
      ExprPtr parent_expr;
      MakeTuplePtr offset_tuple;
      std::string group_id;
      std::vector<ExprPtr> shape_exprs;
      std::vector<ExprPtr> offset_exprs;
    };

    struct RewriteBundle {
      std::vector<StmtPtr> stmts;
    };

    struct LoopDisjointnessCandidate {
      ForStmtPtr loop;
      const std::unordered_set<const Var*>* loop_local_allocs = nullptr;
    };

    std::optional<RewriteBundle> TryRewriteCall(const AssignStmtPtr& call_assign) {
      auto submit = As<Submit>(call_assign->value_);
      auto call = submit ? SubmitToCallView(submit) : As<Call>(call_assign->value_);
      if (!call) return std::nullopt;

      auto callee_name = GetCallFuncName(call);
      auto analysis_it = analyses_.find(callee_name);
      if (analysis_it == analyses_.end()) return std::nullopt;
      auto clone_it = cloned_funcs_.find(callee_name);
      if (clone_it == cloned_funcs_.end()) return std::nullopt;
      auto original_func = program_ ? program_->GetFunction(callee_name) : nullptr;
      if (!original_func) return std::nullopt;

      const auto& analysis = analysis_it->second;
      auto cloned_func = clone_it->second;

      if (analysis.outputs.empty()) return std::nullopt;
      bool disjoint_ok = ProveCallsiteDisjointness(call_assign, call, analysis);
      if (!disjoint_ok) return std::nullopt;

      std::unordered_map<const Var*, ExprPtr> callsite_subst;
      for (size_t i = 0; i < original_func->params_.size() && i < call->args_.size(); ++i) {
        callsite_subst[original_func->params_[i].get()] = call->args_[i];
      }

      std::unordered_map<size_t, SliceBundle> slices_by_out_index;
      std::vector<StmtPtr> stmts;
      stmts.reserve(analysis.outputs.size() * 2 + 2);

      for (const auto& output : analysis.outputs) {
        if (output.out_param_index >= call->args_.size()) return std::nullopt;
        auto out_arg = AsVarLike(call->args_[output.out_param_index]);
        if (!out_arg) return std::nullopt;

        std::vector<ExprPtr> shape_exprs;
        shape_exprs.reserve(output.window_shape.size());
        for (const auto& dim : output.window_shape) {
          auto shape = transform_utils::Substitute(dim, callsite_subst);
          shape = transform_utils::Substitute(shape, scalar_defs_);
          shape_exprs.push_back(arith::Analyzer().Simplify(shape));
        }
        auto shape_tuple = std::make_shared<MakeTuple>(shape_exprs, call_assign->span_);

        std::vector<ExprPtr> offset_exprs;
        offset_exprs.reserve(output.callsite_offsets.size());
        for (const auto& offset : output.callsite_offsets) {
          auto rewritten_offset = transform_utils::Substitute(offset, callsite_subst);
          rewritten_offset = transform_utils::Substitute(rewritten_offset, scalar_defs_);
          offset_exprs.push_back(arith::Analyzer().Simplify(rewritten_offset));
        }
        auto offset_tuple = std::make_shared<MakeTuple>(offset_exprs, call_assign->span_);

        ExprPtr raw_parent_expr = call->args_[output.out_param_index];
        ExprPtr raw_materialized_parent_expr = ResolveLoopInitExpr(raw_parent_expr);
        ExprPtr parent_expr = VisitExpr(raw_parent_expr);
        ExprPtr materialized_parent_expr = VisitExpr(raw_materialized_parent_expr);
        auto parent_var = AsVarLike(materialized_parent_expr);
        if (!parent_var) return std::nullopt;
        auto raw_slice_expr = OpRegistry::GetInstance().Create(
            "tensor.slice", {parent_expr, shape_tuple, offset_tuple}, call_assign->span_);
        auto raw_slice_call = As<Call>(raw_slice_expr);
        if (!raw_slice_call) return std::nullopt;
        const std::string group_id = parent_var->name_hint_ + "__static_window_group";
        const std::string array_name = parent_var->name_hint_ + "__windows";
        std::vector<std::pair<std::string, std::any>> window_get_attrs;
        window_get_attrs.emplace_back(kStaticWindowArrayNameAttr, array_name);
        window_get_attrs.emplace_back(kStaticWindowLoopVarsAttr, std::vector<VarPtr>{});
        window_get_attrs.emplace_back(kStaticWindowLoopStartsAttr, std::vector<ExprPtr>{});
        window_get_attrs.emplace_back(kStaticWindowLoopStepsAttr, std::vector<ExprPtr>{});
        window_get_attrs.emplace_back(kStaticWindowLoopTripsAttr, std::vector<int64_t>{});
        window_get_attrs.emplace_back("static_window_group_id", group_id);
        auto window_get_call = std::make_shared<Call>(
            std::make_shared<Op>("tensor.static_window_get"), std::vector<ExprPtr>{materialized_parent_expr},
            std::vector<std::pair<std::string, std::any>>{}, std::move(window_get_attrs),
            raw_slice_call->GetType(), call_assign->span_);
        auto slice_var = std::make_shared<Var>(out_arg->name_hint_ + "__window", window_get_call->GetType(),
                                               out_arg->span_);
        stmts.push_back(std::make_shared<AssignStmt>(slice_var, window_get_call, call_assign->span_));
        if (planned_windows_) {
          planned_windows_->push_back(
              PlannedWindowGroup{group_id, array_name, parent_var, shape_exprs, offset_exprs, loop_stack_});
        }
        slices_by_out_index.emplace(output.out_param_index, SliceBundle{slice_var, parent_expr, offset_tuple,
                                                                        group_id, shape_exprs, offset_exprs});
        if (writer_records_) {
          group_id_to_parent_[group_id] = parent_var;
        }
      }

      std::vector<ExprPtr> new_args;
      new_args.reserve(call->args_.size());
      for (size_t i = 0; i < call->args_.size(); ++i) {
        auto slice_it = slices_by_out_index.find(i);
        if (slice_it != slices_by_out_index.end()) {
          new_args.push_back(slice_it->second.slice_var);
        } else {
          new_args.push_back(VisitExpr(call->args_[i]));
        }
      }

      auto cloned_gvar = std::make_shared<GlobalVar>(cloned_func->name_);
      const bool is_submit_call = submit != nullptr || IsSubmitCall(call);
      std::vector<TypePtr> result_types = cloned_func->return_types_;
      if (is_submit_call) {
        auto tuple_ty = As<TupleType>(call->GetType());
        if (!tuple_ty || tuple_ty->types_.size() != result_types.size() + 1) return std::nullopt;
        result_types.push_back(tuple_ty->types_.back());
      }
      TypePtr new_return_type =
          result_types.size() == 1 ? result_types[0] : std::make_shared<TupleType>(result_types);

      auto new_attrs = RewriteCallAttrs(call, analysis, slices_by_out_index);
      ExprPtr new_dispatch_expr;
      if (submit) {
        auto new_deps = ExtractSubmitDeps(&new_attrs);
        std::optional<ExprPtr> new_core_num = submit->core_num_;
        if (new_core_num.has_value()) {
          new_core_num = VisitExpr(*new_core_num);
        }
        new_dispatch_expr = std::make_shared<Submit>(
            cloned_gvar, new_args, std::move(new_deps), submit->kwargs_, std::move(new_attrs),
            new_return_type, call->span_, std::move(new_core_num), submit->sync_start_);
      } else {
        new_dispatch_expr = std::make_shared<Call>(cloned_gvar, new_args, call->kwargs_, std::move(new_attrs),
                                                   new_return_type, call->span_);
      }
      auto tmp_result_var = std::make_shared<Var>(call_assign->var_->name_hint_ + "__windowed",
                                                  new_return_type, call_assign->var_->span_);
      stmts.push_back(std::make_shared<AssignStmt>(tmp_result_var, new_dispatch_expr, call_assign->span_));

      if (!is_submit_call && analysis.outputs.size() == 1 && result_types.size() == 1) {
        const auto& output = analysis.outputs[0];
        const auto& slice_bundle = slices_by_out_index.at(output.out_param_index);
        auto assemble_call = OpRegistry::GetInstance().Create(
            "tensor.assemble", {slice_bundle.parent_expr, ExprPtr(tmp_result_var), slice_bundle.offset_tuple},
            call_assign->span_);
        stmts.push_back(std::make_shared<AssignStmt>(call_assign->var_, assemble_call, call_assign->span_));

        RewriteBundle bundle;
        bundle.stmts = std::move(stmts);
        return bundle;
      }

      std::vector<ExprPtr> assembled_result_exprs(result_types.size());
      std::vector<StmtPtr> tail_stmts;
      tail_stmts.reserve(analysis.outputs.size() * 2 + 1);

      std::unordered_map<size_t, VarPtr> tuple_items;
      for (const auto& output : analysis.outputs) {
        auto get_item = std::make_shared<TupleGetItemExpr>(
            tmp_result_var, static_cast<int>(output.return_index), call_assign->span_);
        auto item_var = std::make_shared<Var>(
            call_assign->var_->name_hint_ + "__windowed_" + std::to_string(output.return_index),
            result_types[output.return_index], call_assign->var_->span_);
        tail_stmts.push_back(std::make_shared<AssignStmt>(item_var, get_item, call_assign->span_));

        const auto& slice_bundle = slices_by_out_index.at(output.out_param_index);
        auto assemble_call = OpRegistry::GetInstance().Create(
            "tensor.assemble", {slice_bundle.parent_expr, ExprPtr(item_var), slice_bundle.offset_tuple},
            call_assign->span_);
        auto parent_type = slice_bundle.parent_expr->GetType();
        auto assembled_var = std::make_shared<Var>(
            call_assign->var_->name_hint_ + "__assembled_" + std::to_string(output.return_index), parent_type,
            call_assign->var_->span_);
        tail_stmts.push_back(std::make_shared<AssignStmt>(assembled_var, assemble_call, call_assign->span_));
        assembled_result_exprs[output.return_index] = assembled_var;
      }

      for (size_t i = 0; i < assembled_result_exprs.size(); ++i) {
        if (!assembled_result_exprs[i]) {
          auto get_item =
              std::make_shared<TupleGetItemExpr>(tmp_result_var, static_cast<int>(i), call_assign->span_);
          auto item_var = std::make_shared<Var>(call_assign->var_->name_hint_ + "__pass_" + std::to_string(i),
                                                result_types[i], call_assign->var_->span_);
          tail_stmts.push_back(std::make_shared<AssignStmt>(item_var, get_item, call_assign->span_));
          assembled_result_exprs[i] = item_var;
        }
      }

      tuple_result_subst_[call_assign->var_.get()] = std::move(assembled_result_exprs);
      stmts.insert(stmts.end(), tail_stmts.begin(), tail_stmts.end());

      if (is_submit_call && writer_records_) {
        int tid_idx = static_cast<int>(result_types.size() - 1);
        auto tid_type = result_types.back();
        auto tid_expr = tuple_result_subst_.at(call_assign->var_.get())[static_cast<size_t>(tid_idx)];
        auto tid_var =
            std::make_shared<Var>(call_assign->var_->name_hint_ + "__tid", tid_type, call_assign->span_);
        stmts.push_back(std::make_shared<AssignStmt>(tid_var, tid_expr, call_assign->span_));

        ForStmtPtr enclosing_static_loop;
        ExprPtr window_index_expr;
        if (!loop_stack_.empty()) {
          auto loop = loop_stack_.back();
          auto trip = GetStaticTripCount(loop);
          auto step = GetConstIntValue(loop->step_);
          if (trip.has_value() && *trip > 0 && step.has_value() && *step == 1) {
            enclosing_static_loop = loop;
            window_index_expr = MakeSub(loop->loop_var_, loop->start_, loop->span_);
            window_index_expr = arith::Analyzer().Simplify(window_index_expr);
          }
        }

        for (const auto& output : analysis.outputs) {
          auto slice_it = slices_by_out_index.find(output.out_param_index);
          if (slice_it == slices_by_out_index.end()) continue;
          auto parent_it = group_id_to_parent_.find(slice_it->second.group_id);
          if (parent_it == group_id_to_parent_.end()) continue;
          VarPtr enclosing_loop_var = enclosing_static_loop ? enclosing_static_loop->loop_var_ : nullptr;
          CanonicalRegion write_region;
          // Compute canonical write region for this output. Use a no-alias
          // tracker because the parent var is the resolved loop-init / out
          // param root and not part of an alias chain here.
          AliasTracker no_alias;
          auto parent_region = CanonicalRegionAnalysis(no_alias, loop_iter_init_subst_, tuple_result_subst_)
                                   .Resolve(parent_it->second);
          if (parent_region.has_value()) {
            auto slice_region = BuildWriteRegionForSlice(*parent_region, slice_it->second.shape_exprs,
                                                         slice_it->second.offset_exprs);
            if (slice_region.has_value()) write_region = *slice_region;
          }
          writer_records_->push_back(WindowWriteRecord{tid_var, slice_it->second.group_id, parent_it->second,
                                                       enclosing_static_loop, enclosing_loop_var,
                                                       window_index_expr, write_region});
        }
      }
      if (!is_submit_call) {
        auto rebuilt_tuple =
            std::make_shared<MakeTuple>(tuple_result_subst_.at(call_assign->var_.get()), call_assign->span_);
        stmts.push_back(std::make_shared<AssignStmt>(call_assign->var_, rebuilt_tuple, call_assign->span_));
      }

      RewriteBundle bundle;
      bundle.stmts = std::move(stmts);
      return bundle;
    }

    static bool IsSubmitCall(const CallPtr& call) {
      auto tuple_ty = As<TupleType>(call->GetType());
      if (!tuple_ty || tuple_ty->types_.empty()) return false;
      auto last = As<ScalarType>(tuple_ty->types_.back());
      return last != nullptr && last->dtype_ == DataType::TASK_ID;
    }

    static std::vector<ExprPtr> ExtractSubmitDeps(std::vector<std::pair<std::string, std::any>>* attrs) {
      std::vector<ExprPtr> deps;
      if (!attrs) return deps;
      std::vector<std::pair<std::string, std::any>> kept_attrs;
      kept_attrs.reserve(attrs->size());
      for (auto& [k, v] : *attrs) {
        if (k != kAttrManualDepEdges) {
          kept_attrs.emplace_back(k, v);
          continue;
        }
        const auto* dep_vars = std::any_cast<std::vector<VarPtr>>(&v);
        if (!dep_vars) continue;
        deps.reserve(deps.size() + dep_vars->size());
        for (const auto& dep : *dep_vars) {
          if (dep) deps.push_back(dep);
        }
      }
      *attrs = std::move(kept_attrs);
      return deps;
    }

    std::vector<std::pair<std::string, std::any>> RewriteCallAttrs(
        const CallPtr& call, const CalleeRewriteAnalysis& analysis,
        const std::unordered_map<size_t, SliceBundle>& slices_by_out_index) const {
      std::vector<std::pair<std::string, std::any>> attrs;
      attrs.reserve(call->attrs_.size());
      for (const auto& [k, v] : call->attrs_) {
        if (k == kAttrArgDirections) continue;
        attrs.emplace_back(k, v);
      }
      for (auto& [k, v] : attrs) {
        if (k != kAttrManualDepEdges) continue;
        const auto* user_deps = std::any_cast<std::vector<VarPtr>>(&v);
        if (!user_deps) break;
        std::vector<VarPtr> rewritten;
        rewritten.reserve(user_deps->size());
        bool changed = false;
        for (const auto& dep : *user_deps) {
          bool replaced = false;
          for (const auto& output : analysis.outputs) {
            auto out_arg = AsVarLike(call->args_[output.out_param_index]);
            if (dep && out_arg && dep.get() == out_arg.get()) {
              rewritten.push_back(slices_by_out_index.at(output.out_param_index).slice_var);
              changed = true;
              replaced = true;
              break;
            }
          }
          if (!replaced) rewritten.push_back(dep);
        }
        if (changed) {
          return WithManualDepEdgesAttr(std::move(attrs), std::move(rewritten));
        }
        break;
      }
      return attrs;
    }

    bool ProveCallsiteDisjointness(const AssignStmtPtr& call_assign, const CallPtr& call,
                                   const CalleeRewriteAnalysis& analysis) const {
      if (while_depth_ > 0) return false;
      std::vector<LoopDisjointnessCandidate> candidate_loops;
      candidate_loops.reserve(sequential_loops_.size());
      for (size_t i = 0; i < sequential_loops_.size(); ++i) {
        const auto& loop = sequential_loops_[i];
        if (!loop) continue;
        const auto* local_allocs = i < loop_local_allocs_.size() ? &loop_local_allocs_[i] : nullptr;
        candidate_loops.push_back(LoopDisjointnessCandidate{loop, local_allocs});
      }
      if (candidate_loops.empty()) return true;

      auto original_func = program_ ? program_->GetFunction(call->op_->name_) : nullptr;
      if (!original_func) return false;

      std::unordered_map<const Var*, ExprPtr> callsite_subst;
      for (size_t i = 0; i < original_func->params_.size() && i < call->args_.size(); ++i) {
        callsite_subst[original_func->params_[i].get()] = call->args_[i];
      }

      for (const auto& output : analysis.outputs) {
        if (output.out_param_index >= original_func->params_.size()) return false;
        if (!ProveOutputDisjoint(candidate_loops, output,
                                 original_func->params_[output.out_param_index].get(), callsite_subst)) {
          return false;
        }
      }
      return true;
    }

    bool ProveOutputDisjoint(const std::vector<LoopDisjointnessCandidate>& loops,
                             const OutputRewriteInfo& output, const Var* output_param,
                             const std::unordered_map<const Var*, ExprPtr>& callsite_subst) const {
      std::unordered_set<size_t> varying_dims_used;
      for (const auto& candidate : loops) {
        auto loop = candidate.loop;
        if (IsOutputParentLocalToLoop(output_param, callsite_subst, candidate.loop_local_allocs)) {
          continue;
        }

        auto trip_count = GetStaticTripCount(loop);
        if (!trip_count.has_value()) return false;
        if (*trip_count <= 1) continue;

        std::optional<size_t> varying_dim;
        for (size_t i = 0; i < output.callsite_offsets.size(); ++i) {
          auto rewritten = transform_utils::Substitute(output.callsite_offsets[i], callsite_subst);
          rewritten = transform_utils::Substitute(rewritten, scalar_defs_);
          auto affine = ParseAffineInLoop(rewritten, loop->loop_var_.get());
          if (!affine.has_value()) return false;
          if (affine->coeff == 0) continue;

          auto extent_ci = As<ConstInt>(output.window_shape[i]);
          auto loop_step = GetConstIntValue(loop->step_);
          if (!extent_ci || !loop_step.has_value()) return false;
          if (varying_dim.has_value()) return false;
          if (varying_dims_used.count(i)) return false;
          if (std::abs(affine->coeff * *loop_step) < extent_ci->value_) return false;
          varying_dim = i;
        }
        if (!varying_dim.has_value()) return false;
        varying_dims_used.insert(*varying_dim);
      }
      return true;
    }

    bool IsOutputParentLocalToLoop(const Var* output_param,
                                   const std::unordered_map<const Var*, ExprPtr>& callsite_subst,
                                   const std::unordered_set<const Var*>* loop_local_allocs) const {
      if (!loop_local_allocs || loop_local_allocs->empty()) return false;

      auto subst_it = callsite_subst.find(output_param);
      if (subst_it == callsite_subst.end()) return false;

      auto parent_expr = ResolveLoopInitExpr(subst_it->second);
      auto parent_var = AsVarLike(parent_expr);
      return parent_var && loop_local_allocs->count(parent_var.get());
    }

    ExprPtr ResolveLoopInitExpr(const ExprPtr& expr) const {
      ExprPtr current = expr;
      std::unordered_set<const Var*> seen;
      while (auto var = AsVarLike(current)) {
        if (!seen.insert(var.get()).second) break;
        auto it = loop_iter_init_subst_.find(var.get());
        if (it == loop_iter_init_subst_.end()) break;
        current = it->second;
      }
      return current;
    }

    ExprPtr VisitExpr_(const TupleGetItemExprPtr& op) override {
      auto tuple_var = AsVarLike(op->tuple_);
      if (tuple_var) {
        auto subst_it = tuple_result_subst_.find(tuple_var.get());
        if (subst_it != tuple_result_subst_.end() && op->index_ >= 0 &&
            static_cast<size_t>(op->index_) < subst_it->second.size()) {
          return VisitExpr(subst_it->second[static_cast<size_t>(op->index_)]);
        }
      }
      return IRMutator::VisitExpr_(op);
    }

    ProgramPtr program_;
    const AnalysisMap& analyses_;
    const std::unordered_map<std::string, FunctionPtr>& cloned_funcs_;
    std::vector<PlannedWindowGroup>* planned_windows_ = nullptr;
    std::vector<WindowWriteRecord>* writer_records_ = nullptr;
    std::unordered_map<std::string, VarPtr> group_id_to_parent_;
    std::vector<ForStmtPtr> sequential_loops_;
    std::vector<ForStmtPtr> loop_stack_;
    std::vector<std::unordered_set<const Var*>> loop_local_allocs_;
    std::unordered_map<const Var*, ExprPtr> loop_iter_init_subst_;
    std::unordered_map<const Var*, ExprPtr> scalar_defs_;
    std::unordered_map<const Var*, std::vector<ExprPtr>> tuple_result_subst_;
    int while_depth_ = 0;
  };

  class StaticWindowMaterializer : public IRMutator {
   public:
    explicit StaticWindowMaterializer(std::vector<PlannedWindowGroup> planned_windows)
        : planned_windows_(std::move(planned_windows)) {
      for (auto& group : planned_windows_) {
        groups_by_parent_[group.parent.get()].push_back(&group);
        groups_by_parent_name_[group.parent->name_hint_].push_back(&group);
        groups_by_id_[group.group_id] = &group;
      }
    }

   protected:
    StmtPtr VisitStmt_(const ForStmtPtr& op) override {
      active_loops_.push_back(op->loop_var_.get());
      auto result = IRMutator::VisitStmt_(op);
      active_loops_.pop_back();
      return result;
    }

    StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
      std::vector<StmtPtr> new_stmts;
      new_stmts.reserve(op->stmts_.size() + planned_windows_.size());
      bool changed = false;

      for (const auto& stmt : op->stmts_) {
        auto visited = VisitStmt(stmt);
        if (visited.get() != stmt.get()) changed = true;
        new_stmts.push_back(visited);

        auto original_assign = As<AssignStmt>(stmt);
        if (original_assign) {
          auto precompute_stmts = MakePrecomputeStmtsForParent(original_assign->var_, original_assign->span_);
          if (!precompute_stmts.empty()) {
            new_stmts.insert(new_stmts.end(), precompute_stmts.begin(), precompute_stmts.end());
            changed = true;
          }
        }
      }

      if (!changed) return op;
      return SeqStmts::Flatten(std::move(new_stmts), op->span_);
    }

    ExprPtr VisitExpr_(const CallPtr& op) override {
      if (op->op_->name_ != "tensor.static_window_get") {
        return IRMutator::VisitExpr_(op);
      }
      auto group_id = op->GetAttr<std::string>("static_window_group_id", "");
      auto group_it = groups_by_id_.find(group_id);
      if (group_it == groups_by_id_.end() || !group_it->second) {
        return IRMutator::VisitExpr_(op);
      }
      auto attrs = op->attrs_;
      ReplaceAttr(&attrs, kStaticWindowLoopVarsAttr, group_it->second->emitted_loop_vars);
      ReplaceAttr(&attrs, kStaticWindowLoopStartsAttr, group_it->second->emitted_loop_starts);
      ReplaceAttr(&attrs, kStaticWindowLoopStepsAttr, group_it->second->emitted_loop_steps);
      ReplaceAttr(&attrs, kStaticWindowLoopTripsAttr, group_it->second->emitted_loop_trips);
      return std::make_shared<Call>(op->op_, op->args_, op->kwargs_, std::move(attrs), op->GetType(),
                                    op->span_);
    }

   private:
    std::vector<StmtPtr> MakePrecomputeStmtsForParent(const VarPtr& parent, const Span& span) {
      std::vector<StmtPtr> stmts;
      if (!parent) return stmts;

      std::vector<PlannedWindowGroup*> groups;
      auto ptr_it = groups_by_parent_.find(parent.get());
      if (ptr_it != groups_by_parent_.end()) {
        groups.insert(groups.end(), ptr_it->second.begin(), ptr_it->second.end());
      }
      auto name_it = groups_by_parent_name_.find(parent->name_hint_);
      if (name_it != groups_by_parent_name_.end()) {
        groups.insert(groups.end(), name_it->second.begin(), name_it->second.end());
      }

      for (auto* group : groups) {
        if (!group || materialized_groups_.count(group->group_id)) continue;
        auto precompute = MakePrecomputeStmt(*group, span);
        if (!precompute) continue;
        materialized_groups_.insert(group->group_id);
        stmts.push_back(precompute);
      }
      return stmts;
    }

    static void ReplaceAttr(std::vector<std::pair<std::string, std::any>>* attrs, const std::string& key,
                            std::any value) {
      for (auto& [k, v] : *attrs) {
        if (k == key) {
          v = std::move(value);
          return;
        }
      }
      attrs->emplace_back(key, std::move(value));
    }

    bool IsActiveLoop(const Var* loop_var) const {
      for (const auto* active : active_loops_) {
        if (active == loop_var) return true;
      }
      return false;
    }

    StmtPtr MakePrecomputeStmt(PlannedWindowGroup& group, const Span& span) {
      group.emitted_loop_vars.clear();
      group.emitted_loop_starts.clear();
      group.emitted_loop_steps.clear();
      group.emitted_loop_trips.clear();

      for (const auto& loop : group.loop_stack) {
        if (IsActiveLoop(loop->loop_var_.get())) continue;
        auto trip = GetStaticTripCount(loop);
        if (!trip.has_value() || *trip <= 0) return nullptr;
        group.emitted_loop_vars.push_back(loop->loop_var_);
        group.emitted_loop_starts.push_back(loop->start_);
        group.emitted_loop_steps.push_back(loop->step_);
        group.emitted_loop_trips.push_back(*trip);
      }

      std::vector<std::pair<std::string, std::any>> attrs;
      attrs.emplace_back(kStaticWindowArrayNameAttr, group.array_name);
      attrs.emplace_back(kStaticWindowShapeAttr, group.shape);
      attrs.emplace_back(kStaticWindowOffsetAttr, group.offset);
      attrs.emplace_back(kStaticWindowLoopVarsAttr, group.emitted_loop_vars);
      attrs.emplace_back(kStaticWindowLoopStartsAttr, group.emitted_loop_starts);
      attrs.emplace_back(kStaticWindowLoopStepsAttr, group.emitted_loop_steps);
      attrs.emplace_back(kStaticWindowLoopTripsAttr, group.emitted_loop_trips);
      auto call = std::make_shared<Call>(
          std::make_shared<Op>("tensor.precompute_static_windows"), std::vector<ExprPtr>{group.parent},
          std::vector<std::pair<std::string, std::any>>{}, std::move(attrs), GetUnknownType(), span);
      return std::make_shared<EvalStmt>(call, span);
    }

    std::vector<PlannedWindowGroup> planned_windows_;
    std::unordered_map<const Var*, std::vector<PlannedWindowGroup*>> groups_by_parent_;
    std::unordered_map<std::string, std::vector<PlannedWindowGroup*>> groups_by_parent_name_;
    std::unordered_map<std::string, PlannedWindowGroup*> groups_by_id_;
    std::unordered_set<std::string> materialized_groups_;
    std::vector<const Var*> active_loops_;
  };

  class AliasTracker {
   public:
    void TrackAlias(const VarPtr& result, const VarPtr& source, const CallPtr& producing_call = nullptr) {
      if (!result || !source) return;
      alias_to_source_[result.get()] = source;
      if (producing_call) {
        var_to_call_[result.get()] = producing_call;
      }
    }

    VarPtr CanonicalRoot(const VarPtr& var) const {
      if (!var) return nullptr;
      const Var* current = var.get();
      VarPtr current_ptr = var;
      std::unordered_set<const Var*> seen;
      while (seen.insert(current).second) {
        auto it = alias_to_source_.find(current);
        if (it == alias_to_source_.end()) return current_ptr;
        current = it->second.get();
        current_ptr = it->second;
      }
      return current_ptr;
    }

    const Var* CanonicalRoot(const Var* var) const {
      if (!var) return var;
      std::unordered_set<const Var*> seen;
      while (seen.insert(var).second) {
        auto it = alias_to_source_.find(var);
        if (it == alias_to_source_.end()) return var;
        var = it->second.get();
      }
      return var;
    }

    CallPtr ProducingCall(const Var* var) const {
      if (!var) return nullptr;
      auto it = var_to_call_.find(var);
      return it != var_to_call_.end() ? it->second : nullptr;
    }

   private:
    std::unordered_map<const Var*, VarPtr> alias_to_source_;
    std::unordered_map<const Var*, CallPtr> var_to_call_;
  };

  // ============================================================================
  // CanonicalRegionAnalysis
  //
  // Resolves a tensor expression to a rectilinear region
  //   (root, root_shape, base_linear, dims[])
  // on top of the root's logical row-major layout.
  //
  // Supported input expressions:
  //   * Var            — full region of the variable's tensor type. Walks the
  //                      alias chain (alias_to_source_) to find the storage
  //                      root, and follows loop_iter_init_subst_ for the
  //                      initial value of an IterArg (loop init alias).
  //   * tensor.reshape(parent, new_shape)
  //                   — element-count equivalent contiguous reshape. Inherits
  //                      root / base_linear, rewrites dims from new_shape.
  //   * tensor.slice / tensor.view(parent, shape, offset)
  //                   — keeps root, adds a base_linear offset, dims extents
  //                      from shape, dims strides inherited from parent.
  //   * tensor.static_window_get(parent)
  //                   — same as slice, using the planned shape/offset.
  //   * tensor.assemble(parent, source, offset)
  //                   — alias to parent's full region (the assemble result
  //                      is the parent).
  //   * tuple_get_item  — pass-through (uses tuple_result_subst_).
  //   * MakeTuple / make_tuple result — follow tuple_result_subst_.
  //
  // Returns nullopt for anything it cannot prove (dynamic rank, unknown
  // shape, …). The dependency planner treats unknown as "may overlap" with
  // any same-root writer so we never miss a dep.
  // ============================================================================
  class CanonicalRegionAnalysis {
   public:
    CanonicalRegionAnalysis(const AliasTracker& alias_tracker,
                            const std::unordered_map<const Var*, ExprPtr>& loop_iter_init_subst,
                            const std::unordered_map<const Var*, std::vector<ExprPtr>>& tuple_result_subst)
        : alias_tracker_(alias_tracker),
          loop_iter_init_subst_(loop_iter_init_subst),
          tuple_result_subst_(tuple_result_subst) {}

    std::optional<CanonicalRegion> Resolve(const ExprPtr& expr) const {
      if (!expr) return std::nullopt;
      auto seen_it = memo_.find(expr.get());
      if (seen_it != memo_.end()) return seen_it->second;

      ExprPtr current = expr;
      std::unordered_set<const Expr*> guard;
      while (current && guard.insert(current.get()).second) {
        auto memo_it = memo_.find(current.get());
        if (memo_it != memo_.end()) return memo_it->second;

        if (auto ia = As<IterArg>(current)) {
          auto it = loop_iter_init_subst_.find(ia.get());
          if (it == loop_iter_init_subst_.end() || !it->second) return std::nullopt;
          current = it->second;
          continue;
        }

        if (auto var = AsVarLike(current)) {
          if (auto region = ResolveVar(var)) {
            memo_[expr.get()] = *region;
            memo_[current.get()] = *region;
            return region;
          }
          memo_[expr.get()] = std::nullopt;
          memo_[current.get()] = std::nullopt;
          return std::nullopt;
        }

        if (auto call = As<Call>(current)) {
          if (auto region = ResolveCall(call)) {
            memo_[expr.get()] = *region;
            memo_[current.get()] = *region;
            return region;
          }
          memo_[expr.get()] = std::nullopt;
          memo_[current.get()] = std::nullopt;
          return std::nullopt;
        }

        if (auto tgi = As<TupleGetItemExpr>(current)) {
          auto tuple_var = AsVarLike(tgi->tuple_);
          if (!tuple_var) return std::nullopt;
          auto subst_it = tuple_result_subst_.find(tuple_var.get());
          if (subst_it == tuple_result_subst_.end()) return std::nullopt;
          if (tgi->index_ < 0 || static_cast<size_t>(tgi->index_) >= subst_it->second.size()) {
            return std::nullopt;
          }
          current = subst_it->second[static_cast<size_t>(tgi->index_)];
          continue;
        }

        return std::nullopt;
      }
      return std::nullopt;
    }

   private:
    std::optional<CanonicalRegion> ResolveVar(const VarPtr& var) const {
      if (!var) return std::nullopt;
      VarPtr current = var;
      std::unordered_set<const Var*> seen;
      while (current && seen.insert(current.get()).second) {
        // If this var has a producing call (slice/reshape/assemble/...
        // tracked by the alias tracker), apply the transform on top of
        // the parent region's region.
        auto producing = alias_tracker_.ProducingCall(current.get());
        if (producing) {
          return ResolveCall(producing);
        }
        auto alias_it = alias_tracker_.CanonicalRoot(current);
        if (!alias_it) return std::nullopt;
        if (alias_it.get() == current.get()) {
          return RegionForVarPtr(current);
        }
        current = alias_it;
        auto iter_it = loop_iter_init_subst_.find(current.get());
        if (iter_it != loop_iter_init_subst_.end() && iter_it->second) {
          return Resolve(iter_it->second);
        }
      }
      return std::nullopt;
    }

    std::optional<CanonicalRegion> ResolveCall(const CallPtr& call) const {
      if (!call || !call->op_) return std::nullopt;
      const auto& name = call->op_->name_;
      if (name == "tensor.reshape" && call->args_.size() == 2) {
        auto parent = AsVarLike(call->args_[0]);
        if (!parent) return std::nullopt;
        auto parent_region = Resolve(parent);
        if (!parent_region.has_value()) return std::nullopt;
        auto shape_tuple = As<MakeTuple>(call->args_[1]);
        if (!shape_tuple) return std::nullopt;
        std::vector<ExprPtr> new_shape = shape_tuple->elements_;
        if (new_shape.empty()) return std::nullopt;
        for (const auto& d : new_shape) {
          if (!As<ConstInt>(d)) return std::nullopt;
        }
        auto old_product = ProductConstInt(parent_region->root_shape);
        auto new_product = ProductConstInt(new_shape);
        if (old_product < 0 || new_product < 0 || old_product != new_product) return std::nullopt;
        CanonicalRegion out = *parent_region;
        out.dims = MakeRowMajorDims(new_shape);
        return out;
      }
      if ((name == "tensor.slice" || name == "tensor.view") && call->args_.size() >= 3) {
        auto parent = AsVarLike(call->args_[0]);
        if (!parent) return std::nullopt;
        auto parent_region = Resolve(parent);
        if (!parent_region.has_value()) return std::nullopt;
        auto shape_tuple = As<MakeTuple>(call->args_[1]);
        auto offset_tuple = As<MakeTuple>(call->args_[2]);
        if (!shape_tuple || !offset_tuple) return std::nullopt;
        if (shape_tuple->elements_.size() != parent_region->dims.size()) return std::nullopt;
        if (offset_tuple->elements_.size() != parent_region->dims.size()) return std::nullopt;
        for (const auto& d : shape_tuple->elements_) {
          if (!As<ConstInt>(d)) return std::nullopt;
        }
        CanonicalRegion out;
        out.root = parent_region->root;
        out.root_shape = parent_region->root_shape;
        auto base_linear =
            AddLinearOffset(parent_region->base_linear, parent_region->dims, offset_tuple->elements_);
        if (!base_linear.has_value()) return std::nullopt;
        out.base_linear = *base_linear;
        out.dims.reserve(parent_region->dims.size());
        for (size_t i = 0; i < parent_region->dims.size(); ++i) {
          out.dims.push_back(RegionDim{shape_tuple->elements_[i], parent_region->dims[i].stride});
        }
        return out;
      }
      if (name == "tensor.static_window_get" && call->args_.size() >= 1) {
        auto parent = AsVarLike(call->args_[0]);
        if (!parent) return std::nullopt;
        auto shape_attr = call->GetAttr<std::vector<ExprPtr>>(kStaticWindowShapeAttr, {});
        auto offset_attr = call->GetAttr<std::vector<ExprPtr>>(kStaticWindowOffsetAttr, {});
        if (shape_attr.empty() || offset_attr.empty()) return std::nullopt;
        if (shape_attr.size() != offset_attr.size()) return std::nullopt;
        for (const auto& d : shape_attr) {
          if (!As<ConstInt>(d)) return std::nullopt;
        }
        auto parent_region = Resolve(parent);
        if (!parent_region.has_value()) return std::nullopt;
        if (shape_attr.size() != parent_region->dims.size()) return std::nullopt;
        CanonicalRegion out;
        out.root = parent_region->root;
        out.root_shape = parent_region->root_shape;
        auto base_linear = AddLinearOffset(parent_region->base_linear, parent_region->dims, offset_attr);
        if (!base_linear.has_value()) return std::nullopt;
        out.base_linear = *base_linear;
        out.dims.reserve(parent_region->dims.size());
        for (size_t i = 0; i < parent_region->dims.size(); ++i) {
          out.dims.push_back(RegionDim{shape_attr[i], parent_region->dims[i].stride});
        }
        return out;
      }
      if ((name == "tensor.assemble" || name == "tensor.as_layout" || name == "tensor.set_validshape") &&
          !call->args_.empty()) {
        auto parent = AsVarLike(call->args_[0]);
        if (!parent) return std::nullopt;
        return Resolve(parent);
      }
      return std::nullopt;
    }

    std::optional<CanonicalRegion> RegionForVarPtr(const VarPtr& var) const {
      if (!var) return std::nullopt;
      auto tensor_type = As<TensorType>(var->GetType());
      if (!tensor_type) return std::nullopt;
      if (tensor_type->shape_.empty()) return std::nullopt;
      for (const auto& d : tensor_type->shape_) {
        if (!As<ConstInt>(d)) return std::nullopt;
      }
      CanonicalRegion out;
      out.root = var;
      out.root_shape = tensor_type->shape_;
      out.base_linear = std::make_shared<ConstInt>(0, DataType::INDEX, var->span_);
      out.dims = MakeRowMajorDims(tensor_type->shape_);
      return out;
    }

    static std::vector<RegionDim> MakeRowMajorDims(const std::vector<ExprPtr>& shape) {
      std::vector<RegionDim> dims;
      dims.reserve(shape.size());
      int64_t product = 1;
      for (auto it = shape.rbegin(); it != shape.rend(); ++it) {
        auto ci = As<ConstInt>(*it);
        if (!ci) return {};
        dims.insert(dims.begin(),
                    RegionDim{*it, std::make_shared<ConstInt>(product, DataType::INDEX, (*it)->span_)});
        product *= ci->value_;
      }
      return dims;
    }

    static int64_t ProductConstInt(const std::vector<ExprPtr>& exprs) {
      int64_t product = 1;
      for (const auto& e : exprs) {
        auto ci = As<ConstInt>(e);
        if (!ci) return -1;
        product *= ci->value_;
      }
      return product;
    }

    static std::optional<ExprPtr> AddLinearOffset(const ExprPtr& base,
                                                  const std::vector<RegionDim>& parent_dims,
                                                  const std::vector<ExprPtr>& offset) {
      if (!base) return std::nullopt;
      auto base_ci = As<ConstInt>(base);
      if (!base_ci) return std::nullopt;
      int64_t total = base_ci->value_;
      bool touched = false;
      for (size_t i = 0; i < parent_dims.size() && i < offset.size(); ++i) {
        auto off_ci = As<ConstInt>(offset[i]);
        auto stride_ci = As<ConstInt>(parent_dims[i].stride);
        if (!off_ci || !stride_ci) return std::nullopt;
        if (off_ci->value_ == 0) continue;
        int64_t contribution = off_ci->value_ * stride_ci->value_;
        total += contribution;
        touched = true;
      }
      if (!touched) return base;
      return std::make_shared<ConstInt>(total, DataType::INDEX, base->span_);
    }

    const AliasTracker& alias_tracker_;
    const std::unordered_map<const Var*, ExprPtr>& loop_iter_init_subst_;
    const std::unordered_map<const Var*, std::vector<ExprPtr>>& tuple_result_subst_;
    mutable std::unordered_map<const Expr*, std::optional<CanonicalRegion>> memo_;
  };

  // ============================================================================
  // RegionOverlapper
  //
  // Conservative may-overlap test for two regions sharing the same root.
  //   * If dim strides differ → "may overlap" (cannot compare intervals).
  //   * If at least one dim's offset/extent proves the intervals
  //     [off, off+extent) and [other_off, other_off+other_extent) are
  //     disjoint (via ConstInt arithmetic) → "disjoint".
  //   * Otherwise → "may overlap".
  // Only ConstInt extents/offsets are handled in v1; loop-affine
  // expressions fall through to "may overlap" so the planner errs on the
  // side of inserting a dep.
  // ============================================================================
  static bool RegionsMayOverlap(const CanonicalRegion& a, const CanonicalRegion& b) {
    if (a.root.get() != b.root.get()) return false;
    if (a.dims.size() != b.dims.size()) return true;
    if (a.dims.size() != a.root_shape.size() || b.dims.size() != b.root_shape.size()) return true;
    for (size_t i = 0; i < a.dims.size(); ++i) {
      auto a_stride = As<ConstInt>(a.dims[i].stride);
      auto b_stride = As<ConstInt>(b.dims[i].stride);
      if (!a_stride || !b_stride || a_stride->value_ != b_stride->value_) return true;
    }
    auto a_base = As<ConstInt>(a.base_linear);
    auto b_base = As<ConstInt>(b.base_linear);
    if (!a_base || !b_base) return true;

    auto coord_for_dim = [](int64_t base, const RegionDim& dim,
                            const ExprPtr& root_dim) -> std::optional<int64_t> {
      auto stride = As<ConstInt>(dim.stride);
      auto root_extent = As<ConstInt>(root_dim);
      if (!stride || !root_extent || stride->value_ <= 0 || root_extent->value_ <= 0) return std::nullopt;
      return (base / stride->value_) % root_extent->value_;
    };

    for (size_t i = 0; i < a.dims.size(); ++i) {
      auto a_extent = As<ConstInt>(a.dims[i].extent);
      auto b_extent = As<ConstInt>(b.dims[i].extent);
      if (!a_extent || !b_extent) return true;
      auto a_coord = coord_for_dim(a_base->value_, a.dims[i], a.root_shape[i]);
      auto b_coord = coord_for_dim(b_base->value_, b.dims[i], b.root_shape[i]);
      if (!a_coord.has_value() || !b_coord.has_value()) return true;
      int64_t a_start = *a_coord;
      int64_t a_end = a_start + a_extent->value_;
      int64_t b_start = *b_coord;
      int64_t b_end = b_start + b_extent->value_;
      if (a_end <= b_start || b_end <= a_start) return false;
    }
    return true;
  }

  // True if the region's extent matches the parent root's shape on every
  // dim and base_linear is zero. Used to fast-path "full read" — for a
  // full-region reader we keep every pending writer dep.
  static bool IsFullRegion(const CanonicalRegion& region, const Var* canonical) {
    if (region.root.get() != canonical) return false;
    if (region.dims.size() != region.root_shape.size()) return false;
    auto base = As<ConstInt>(region.base_linear);
    if (!base || base->value_ != 0) return false;
    for (size_t i = 0; i < region.dims.size(); ++i) {
      auto ext = As<ConstInt>(region.dims[i].extent);
      auto root = As<ConstInt>(region.root_shape[i]);
      if (!ext || !root || ext->value_ != root->value_) return false;
    }
    return true;
  }

  static std::optional<CanonicalRegion> BuildWriteRegionForSlice(const CanonicalRegion& parent_region,
                                                                 const std::vector<ExprPtr>& shape,
                                                                 const std::vector<ExprPtr>& offset) {
    if (shape.size() != parent_region.dims.size()) return std::nullopt;
    if (offset.size() != parent_region.dims.size()) return std::nullopt;
    for (const auto& d : shape) {
      if (!As<ConstInt>(d)) return std::nullopt;
    }
    CanonicalRegion out = parent_region;
    out.dims.clear();
    out.dims.reserve(parent_region.dims.size());
    for (size_t i = 0; i < parent_region.dims.size(); ++i) {
      out.dims.push_back(RegionDim{shape[i], parent_region.dims[i].stride});
    }
    auto base_ci = As<ConstInt>(parent_region.base_linear);
    if (!base_ci) return std::nullopt;
    int64_t total = base_ci->value_;
    bool touched = false;
    for (size_t i = 0; i < parent_region.dims.size() && i < offset.size(); ++i) {
      auto off_ci = As<ConstInt>(offset[i]);
      auto stride_ci = As<ConstInt>(parent_region.dims[i].stride);
      if (!off_ci || !stride_ci) return std::nullopt;
      if (off_ci->value_ == 0) continue;
      total += off_ci->value_ * stride_ci->value_;
      touched = true;
    }
    if (touched || base_ci) {
      out.base_linear = std::make_shared<ConstInt>(total, DataType::INDEX, parent_region.base_linear->span_);
    } else {
      out.base_linear = parent_region.base_linear;
    }
    return out;
  }

  class YieldAppender : public IRMutator {
   public:
    explicit YieldAppender(std::vector<ExprPtr> values) : values_(std::move(values)) {}

    StmtPtr VisitStmt_(const YieldStmtPtr& op) override {
      auto new_values = op->value_;
      new_values.insert(new_values.end(), values_.begin(), values_.end());
      return std::make_shared<YieldStmt>(std::move(new_values), op->span_);
    }

   private:
    std::vector<ExprPtr> values_;
  };

  class WindowDependencyPlanner : public IRMutator {
   public:
    WindowDependencyPlanner(const ProgramPtr& program, std::vector<WindowWriteRecord> writer_records)
        : program_(program), writer_records_(std::move(writer_records)) {
      for (const auto& record : writer_records_) {
        if (record.task_id_var) {
          writer_tid_set_.insert(record.task_id_var.get());
          tid_to_record_[record.task_id_var.get()] = &record;
        }
        if (record.enclosing_loop) {
          records_by_loop_[record.enclosing_loop.get()].push_back(&record);
        }
        if (record.enclosing_loop_var) {
          records_by_loop_var_[record.enclosing_loop_var.get()].push_back(&record);
        }
      }
    }

   protected:
    StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
      auto saved_tids = defined_tids_;
      auto saved_available_deps = available_parent_deps_;
      std::vector<StmtPtr> new_stmts;
      new_stmts.reserve(op->stmts_.size() * 2);
      bool changed = false;

      for (const auto& stmt : op->stmts_) {
        auto assign = As<AssignStmt>(stmt);

        if (assign) {
          TrackAliasFromAssign(assign);
        }

        if (assign && writer_tid_set_.count(assign->var_.get())) {
          defined_tids_.insert(assign->var_.get());
        }

        if (assign) {
          auto rewritten = MaybeInsertBarrier(assign);
          if (rewritten.size() > 1) {
            changed = true;
            for (const auto& s : rewritten) new_stmts.push_back(VisitStmt(s));
            continue;
          }
        }

        auto for_stmt = As<ForStmt>(stmt);
        if (for_stmt) {
          auto rewritten_for = MaybeTransformForStmt(for_stmt);
          if (rewritten_for.has_value()) {
            changed = true;
            for (const auto& s : rewritten_for.value()) new_stmts.push_back(s);
            continue;
          }
        }

        auto visited = VisitStmt(stmt);
        changed = changed || visited.get() != stmt.get();
        new_stmts.push_back(visited);

        if (assign) {
          PropagateAvailableDeps(assign);
        }
      }

      defined_tids_ = std::move(saved_tids);
      available_parent_deps_ = std::move(saved_available_deps);
      if (!changed) return op;
      return SeqStmts::Flatten(std::move(new_stmts), op->span_);
    }

    StmtPtr VisitStmt_(const ForStmtPtr& op) override {
      auto saved_tids = defined_tids_;
      auto saved_available_deps = available_parent_deps_;
      auto visited = IRMutator::VisitStmt_(op);
      defined_tids_ = std::move(saved_tids);
      available_parent_deps_ = std::move(saved_available_deps);
      return visited;
    }

    StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
      auto saved_tids = defined_tids_;
      auto visited = IRMutator::VisitStmt_(op);
      defined_tids_ = std::move(saved_tids);
      return visited;
    }

    ExprPtr VisitExpr_(const CallPtr& op) override {
      if (IsTrackedTensorAliasOp(op->op_->name_) && !op->args_.empty()) {
        auto result_var = current_assign_var_;
        auto parent_var = AsVarLike(op->args_[0]);
        if (result_var && parent_var) {
          alias_tracker_.TrackAlias(result_var, parent_var, op);
        }
      }
      return IRMutator::VisitExpr_(op);
    }

   private:
    void TrackAliasFromAssign(const AssignStmtPtr& assign) {
      current_assign_var_ = assign->var_;
      if (auto source_var = AsVarLike(assign->value_)) {
        if (As<TensorType>(assign->var_->GetType()) && As<TensorType>(source_var->GetType())) {
          alias_tracker_.TrackAlias(assign->var_, source_var);
        }
        return;
      }
      auto call = As<Call>(assign->value_);
      if (!call || !call->op_) return;
      const auto& name = call->op_->name_;
      if (IsTrackedTensorAliasOp(name) && !call->args_.empty()) {
        auto parent_var = AsVarLike(call->args_[0]);
        if (parent_var) {
          alias_tracker_.TrackAlias(assign->var_, parent_var, call);
        }
      }
    }

    void PropagateAvailableDeps(const AssignStmtPtr& assign) {
      auto it = tid_to_record_.find(assign->var_.get());
      if (it == tid_to_record_.end()) return;
      auto* record = it->second;
      if (!record->parent) return;
      auto parent_root = alias_tracker_.CanonicalRoot(record->parent);
      if (parent_root) {
        available_parent_deps_[parent_root.get()].push_back(record->task_id_var);
      }
    }

    std::optional<std::vector<StmtPtr>> MaybeTransformForStmt(const ForStmtPtr& op) {
      auto loop_var = op->loop_var_;
      auto trip = GetStaticTripCount(op);

      std::vector<const WindowWriteRecord*> loop_tids;
      std::unordered_set<const WindowWriteRecord*> seen_loop_records;
      auto by_loop = records_by_loop_.find(op.get());
      if (by_loop != records_by_loop_.end()) {
        for (auto* record : by_loop->second) {
          if (seen_loop_records.insert(record).second) loop_tids.push_back(record);
        }
      }
      if (op->loop_var_) {
        auto by_loop_var = records_by_loop_var_.find(op->loop_var_.get());
        if (by_loop_var != records_by_loop_var_.end()) {
          for (auto* record : by_loop_var->second) {
            if (seen_loop_records.insert(record).second) loop_tids.push_back(record);
          }
        }
      }
      loop_tids.erase(std::remove_if(loop_tids.begin(), loop_tids.end(),
                                     [](const WindowWriteRecord* record) {
                                       return !record || !record->parent || !record->window_index_expr;
                                     }),
                      loop_tids.end());

      TrackLoopIterArgAliases(op);

      if (loop_tids.empty() || !trip.has_value() || *trip <= 0) return std::nullopt;

      auto step = GetConstIntValue(op->step_);
      if (!step.has_value() || *step != 1) return std::nullopt;

      struct CarryGroup {
        VarPtr parent_root;
        ExprPtr window_index_expr;
        std::vector<const Var*> tid_vars;
      };
      std::vector<CarryGroup> carry_groups;
      carry_groups.reserve(loop_tids.size());
      for (auto* record : loop_tids) {
        auto parent_root = alias_tracker_.CanonicalRoot(record->parent);
        if (!parent_root) parent_root = record->parent;
        carry_groups.push_back(
            CarryGroup{parent_root, record->window_index_expr, {record->task_id_var.get()}});
      }

      if (carry_groups.empty()) return std::nullopt;

      std::vector<StmtPtr> pre_loop_stmts;
      auto new_iter_args = op->iter_args_;
      std::vector<VarPtr> new_return_vars = op->return_vars_;
      StmtPtr new_body = VisitStmt(op->body_);
      TrackLoopReturnAliases(op, new_body);
      std::string loop_var_name = loop_var ? loop_var->name_hint_ : "i";
      int64_t trip_count = *trip;

      struct ArrayCarry {
        VarPtr init_var;
        VarPtr iter_var;
        VarPtr result_var;
        VarPtr update_var;
        VarPtr parent_root;
        std::vector<const Var*> tid_vars;
      };
      std::vector<ArrayCarry> carries;

      auto array_type = std::make_shared<ArrayType>(DataType::TASK_ID, trip_count);

      int carry_idx = 0;
      for (auto& group : carry_groups) {
        std::string suffix = "__tids_" + std::to_string(carry_idx++);

        auto init_var = std::make_shared<Var>(loop_var_name + suffix + "_init", array_type, op->span_);
        auto extent = std::make_shared<ConstInt>(trip_count, DataType::INDEX, op->span_);
        std::vector<ExprPtr> create_args{extent};
        std::vector<std::pair<std::string, std::any>> create_kwargs;
        create_kwargs.emplace_back("dtype", DataType::TASK_ID);
        auto create_call = std::make_shared<Call>(
            OpRegistry::GetInstance().GetOp("array.create"), std::move(create_args), std::move(create_kwargs),
            std::vector<std::pair<std::string, std::any>>{}, array_type, op->span_);
        pre_loop_stmts.push_back(std::make_shared<AssignStmt>(init_var, create_call, op->span_));

        auto iter_var = std::make_shared<IterArg>(loop_var_name + suffix + "_iter", array_type,
                                                  ExprPtr(init_var), op->span_);

        auto result_var = std::make_shared<Var>(loop_var_name + suffix + "_final", array_type, op->span_);

        new_iter_args.push_back(iter_var);
        new_return_vars.push_back(result_var);

        carries.push_back(
            {init_var, iter_var, result_var, iter_var, group.parent_root, std::move(group.tid_vars)});
      }

      std::unordered_map<const Var*, size_t> tid_to_carry_idx;
      std::unordered_set<const Var*> all_tid_vars_in_loop;
      for (size_t ci = 0; ci < carries.size(); ++ci) {
        for (auto* tid_var : carries[ci].tid_vars) {
          tid_to_carry_idx[tid_var] = ci;
          all_tid_vars_in_loop.insert(tid_var);
        }
      }

      ExprPtr window_index_expr;
      if (!loop_tids.empty()) {
        window_index_expr = loop_tids[0]->window_index_expr;
        for (auto* record : loop_tids) {
          if (record->window_index_expr) {
            window_index_expr = record->window_index_expr;
            break;
          }
        }
      }

      auto rewrite_body_with_tid_updates = [&](const StmtPtr& body) -> StmtPtr {
        class TidUpdateRewriter : public IRMutator {
         public:
          TidUpdateRewriter(std::vector<ArrayCarry>* carries,
                            const std::unordered_map<const Var*, size_t>& tid_to_carry,
                            const std::unordered_set<const Var*>& all_tids, const ExprPtr& idx)
              : carries_(carries), tid_to_carry_(tid_to_carry), all_tids_(all_tids), idx_(idx) {}

          StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
            if (!all_tids_.count(op->var_.get())) {
              return IRMutator::VisitStmt_(op);
            }
            auto it = tid_to_carry_.find(op->var_.get());
            if (it == tid_to_carry_.end()) {
              return IRMutator::VisitStmt_(op);
            }
            size_t ci = it->second;
            auto& carry = (*carries_)[ci];
            auto original_assign = IRMutator::VisitStmt_(op);

            auto current_array = ExprPtr(carry.iter_var);
            if (carry_updated_in_iteration_.count(ci)) {
              current_array = ExprPtr(carry.update_var);
            }
            carry_updated_in_iteration_.insert(ci);

            auto update_var = std::make_shared<Var>(
                carry.iter_var->name_hint_ + "_updated_" + std::to_string(update_counter_++),
                carry.iter_var->GetType(), op->span_);
            auto update_call = std::make_shared<Call>(OpRegistry::GetInstance().GetOp("array.update_element"),
                                                      std::vector<ExprPtr>{current_array, idx_, op->var_},
                                                      std::vector<std::pair<std::string, std::any>>{},
                                                      std::vector<std::pair<std::string, std::any>>{},
                                                      carry.iter_var->GetType(), op->span_);
            auto update_assign = std::make_shared<AssignStmt>(update_var, update_call, op->span_);
            carry.update_var = update_var;
            return SeqStmts::Flatten({original_assign, update_assign}, op->span_);
          }

          StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
            std::vector<StmtPtr> new_stmts;
            bool changed = false;
            for (const auto& stmt : op->stmts_) {
              auto visited = VisitStmt(stmt);
              changed = changed || visited.get() != stmt.get();
              new_stmts.push_back(visited);
            }
            if (!changed) return op;
            return SeqStmts::Flatten(std::move(new_stmts), op->span_);
          }

         private:
          std::vector<ArrayCarry>* carries_;
          const std::unordered_map<const Var*, size_t>& tid_to_carry_;
          const std::unordered_set<const Var*>& all_tids_;
          ExprPtr idx_;
          std::unordered_set<size_t> carry_updated_in_iteration_;
          int update_counter_ = 0;
        };
        return TidUpdateRewriter(&carries, tid_to_carry_idx, all_tid_vars_in_loop, window_index_expr)
            .VisitStmt(body);
      };

      new_body = rewrite_body_with_tid_updates(new_body);

      auto existing_yield = transform_utils::GetLastYieldStmt(new_body);
      if (existing_yield) {
        std::vector<ExprPtr> yield_values;
        yield_values.reserve(carries.size());
        for (auto& carry : carries) {
          yield_values.push_back(ExprPtr(carry.update_var));
        }
        new_body = YieldAppender(std::move(yield_values)).VisitStmt(new_body);
      } else {
        std::vector<ExprPtr> yield_values;
        for (auto& carry : carries) {
          yield_values.push_back(ExprPtr(carry.update_var));
        }
        auto yield_stmt = std::make_shared<YieldStmt>(std::move(yield_values), op->span_);
        auto body_stmts = FlattenToStmts(new_body);
        body_stmts.push_back(yield_stmt);
        new_body = SeqStmts::Flatten(std::move(body_stmts), op->span_);
      }

      auto new_for = std::make_shared<ForStmt>(op->loop_var_, op->start_, op->stop_, op->step_, new_iter_args,
                                               new_body, new_return_vars, op->span_, op->kind_,
                                               op->chunk_config_, op->attrs_, op->leading_comments_);
      pre_loop_stmts.push_back(new_for);

      for (auto& carry : carries) {
        available_parent_deps_[carry.parent_root.get()].push_back(carry.result_var);
        defined_tids_.insert(carry.result_var.get());
      }

      return pre_loop_stmts;
    }

    bool IsTensorOp(const std::string& name) const {
      return name == "tensor.assemble" || name == "tensor.slice" || name == "tensor.static_window_get" ||
             name == "tensor.precompute_static_windows" || name == "tensor.create" || name == "tensor.full" ||
             name == "tensor.reshape" || name == "tensor.as_layout" || name == "tensor.set_validshape" ||
             name == "array.create" || name == "array.update_element" || name == "array.get_element";
    }

    static bool IsTrackedTensorAliasOp(const std::string& name) {
      return name == "tensor.reshape" || name == "tensor.assemble" || name == "tensor.slice" ||
             name == "tensor.static_window_get" || name == "tensor.as_layout" ||
             name == "tensor.set_validshape";
    }

    void TrackLoopIterArgAliases(const ForStmtPtr& op) {
      for (size_t i = 0; i < op->iter_args_.size() && i < op->return_vars_.size(); ++i) {
        auto iter_arg = op->iter_args_[i];
        if (!iter_arg || !iter_arg->initValue_) continue;
        if (!As<TensorType>(iter_arg->GetType())) continue;
        auto init_var = AsVarLike(iter_arg->initValue_);
        if (!init_var) continue;
        alias_tracker_.TrackAlias(iter_arg, init_var);
      }
    }

    void TrackLoopReturnAliases(const ForStmtPtr& op, const StmtPtr& body) {
      auto yield = transform_utils::GetLastYieldStmt(body);
      if (!yield) return;
      std::unordered_map<const Var*, ExprPtr> loop_iter_init;
      for (const auto& iter_arg : op->iter_args_) {
        if (iter_arg && iter_arg->initValue_) loop_iter_init[iter_arg.get()] = iter_arg->initValue_;
      }
      std::unordered_map<const Var*, std::vector<ExprPtr>> empty_tuple_subst;
      CanonicalRegionAnalysis region_analysis(alias_tracker_, loop_iter_init, empty_tuple_subst);
      for (size_t i = 0; i < op->iter_args_.size() && i < op->return_vars_.size() && i < yield->value_.size();
           ++i) {
        auto iter_arg = op->iter_args_[i];
        auto return_var = op->return_vars_[i];
        if (!iter_arg || !return_var || !iter_arg->initValue_) continue;
        if (!As<TensorType>(return_var->GetType())) continue;
        auto init_var = AsVarLike(iter_arg->initValue_);
        if (!init_var) continue;
        auto init_root = alias_tracker_.CanonicalRoot(init_var);
        auto yielded_region = region_analysis.Resolve(yield->value_[i]);
        if (init_root && yielded_region.has_value() && yielded_region->root.get() == init_root.get()) {
          alias_tracker_.TrackAlias(return_var, init_root);
        }
      }
    }

    bool IsReaderArg(const Var* arg_var, const FunctionPtr& callee, size_t arg_index) const {
      if (!callee || arg_index >= callee->param_directions_.size()) {
        return false;
      }
      auto dir = callee->param_directions_[arg_index];
      return dir == ParamDirection::In || dir == ParamDirection::InOut;
    }

    std::vector<StmtPtr> MaybeInsertBarrier(const AssignStmtPtr& assign) {
      auto submit = As<Submit>(assign->value_);
      auto call = submit ? SubmitToCallView(submit) : As<Call>(assign->value_);
      if (!call) return {assign};
      auto gvar = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
      if (!gvar) return {assign};

      std::string callee_name = gvar->name_;
      if (callee_name == "system.task_dummy") return {assign};
      if (IsTensorOp(callee_name)) return {assign};

      auto callee = program_ ? program_->GetFunction(callee_name) : nullptr;

      // Reader-side region analysis: for each tensor read arg, compute its
      // canonical region so we can prove disjointness against pending writers.
      // The planner only uses alias_tracker_ here — iter-arg init / tuple
      // substitutions live in OrchRewriter's stack and are not visible to a
      // sibling pass; in v1 the reader is always a fresh tensor var / slice
      // call (the only test shapes for the indexer-like case), so the
      // pass-local alias chain is enough.
      std::unordered_map<const Var*, ExprPtr> empty_iter_init;
      std::unordered_map<const Var*, std::vector<ExprPtr>> empty_tuple_subst;
      CanonicalRegionAnalysis region_analysis(alias_tracker_, empty_iter_init, empty_tuple_subst);

      std::vector<VarPtr> overlapping_deps;
      std::unordered_set<const Var*> overlapping_dep_set;
      std::unordered_set<const Var*> roots_to_summarize;
      for (size_t i = 0; i < call->args_.size(); ++i) {
        auto arg_var = AsVarLike(call->args_[i]);
        if (!arg_var) continue;
        if (callee && !IsReaderArg(arg_var.get(), callee, i)) continue;
        const Var* canonical = alias_tracker_.CanonicalRoot(arg_var.get());
        if (!canonical) canonical = arg_var.get();

        auto dep_it = available_parent_deps_.find(canonical);
        if (dep_it == available_parent_deps_.end()) continue;

        std::optional<CanonicalRegion> reader_region = region_analysis.Resolve(arg_var);
        // If the reader's region is unknown / not provable, conservatively
        // assume overlap with every pending writer.
        bool reader_is_full = reader_region.has_value() && IsFullRegion(*reader_region, canonical);
        bool summarize_root = reader_is_full || !reader_region.has_value();
        bool root_has_overlap = false;

        for (const auto& dep : dep_it->second) {
          if (!defined_tids_.count(dep.get())) continue;
          bool overlaps = false;
          if (reader_is_full) {
            overlaps = true;
          } else if (!reader_region.has_value()) {
            overlaps = true;
          } else {
            // Region known and not provably full: keep only writers whose
            // region may overlap the reader's.
            auto writer_it = tid_to_record_.find(dep.get());
            if (writer_it == tid_to_record_.end() || !writer_it->second ||
                writer_it->second->write_region.root == nullptr) {
              overlaps = true;
            } else {
              overlaps = RegionsMayOverlap(*reader_region, writer_it->second->write_region);
            }
          }
          if (overlaps && overlapping_dep_set.insert(dep.get()).second) {
            overlapping_deps.push_back(dep);
            root_has_overlap = true;
          }
        }
        if (summarize_root && root_has_overlap) {
          roots_to_summarize.insert(canonical);
        }
      }
      if (overlapping_deps.empty()) return {assign};

      auto span = assign->span_;
      auto tid_type = std::make_shared<ScalarType>(DataType::TASK_ID);
      auto barrier_var =
          std::make_shared<Var>(assign->var_->name_hint_ + "__window_barrier_tid", tid_type, span);
      std::vector<std::pair<std::string, std::any>> barrier_attrs;
      barrier_attrs.emplace_back(kAttrDummyTask, true);
      barrier_attrs.emplace_back(kAttrManualDepEdges, std::move(overlapping_deps));
      auto barrier_call = std::make_shared<Call>(
          OpRegistry::GetInstance().GetOp("system.task_dummy"), std::vector<ExprPtr>{},
          std::vector<std::pair<std::string, std::any>>{}, std::move(barrier_attrs), tid_type, span);
      auto barrier_stmt = std::make_shared<AssignStmt>(barrier_var, barrier_call, span);

      ExprPtr new_dispatch_expr;
      if (submit) {
        std::vector<ExprPtr> new_deps = submit->deps_;
        new_deps.push_back(barrier_var);
        new_dispatch_expr = std::make_shared<Submit>(submit->op_, submit->args_, std::move(new_deps),
                                                     submit->kwargs_, submit->attrs_, submit->GetType(),
                                                     submit->span_, submit->core_num_, submit->sync_start_);
      } else {
        new_dispatch_expr = std::make_shared<Call>(call->op_, call->args_, call->kwargs_,
                                                   WithManualDepEdgesAttr(call->attrs_, {barrier_var}),
                                                   call->GetType(), call->span_);
      }
      auto new_assign = std::make_shared<AssignStmt>(assign->var_, new_dispatch_expr, assign->span_);

      defined_tids_.insert(barrier_var.get());
      for (const auto* root : roots_to_summarize) {
        available_parent_deps_[root] = {barrier_var};
      }

      return {barrier_stmt, new_assign};
    }

    ProgramPtr program_;
    std::vector<WindowWriteRecord> writer_records_;
    std::unordered_set<const Var*> writer_tid_set_;
    std::unordered_set<const Var*> defined_tids_;
    std::unordered_map<const Var*, const WindowWriteRecord*> tid_to_record_;
    std::unordered_map<const ForStmt*, std::vector<const WindowWriteRecord*>> records_by_loop_;
    std::unordered_map<const Var*, std::vector<const WindowWriteRecord*>> records_by_loop_var_;
    std::unordered_map<const Var*, std::vector<VarPtr>> available_parent_deps_;
    AliasTracker alias_tracker_;
    VarPtr current_assign_var_;

    friend class BodyRewriter;
  };

  struct FinalStoreInfo {
    size_t return_index;
    std::vector<ExprPtr> window_shape;
    std::vector<ExprPtr> offsets;
  };

  struct AggregateWindowInfo {
    size_t return_index;
    std::vector<ExprPtr> window_shape;
    std::vector<ExprPtr> base_offsets;
    std::vector<ExprPtr> local_offsets;
    size_t iter_arg_index;
  };

  static std::optional<size_t> FindReturnIndexForOutParam(const FunctionPtr& func, size_t out_param_index) {
    if (!func || out_param_index >= func->params_.size()) return std::nullopt;
    auto body_stmts = FlattenToStmts(func->body_);
    ReturnStmtPtr ret_stmt;
    for (const auto& stmt : body_stmts) {
      if (auto ret = As<ReturnStmt>(stmt)) {
        ret_stmt = ret;
        break;
      }
    }
    if (!ret_stmt) return std::nullopt;

    const auto* out_param = func->params_[out_param_index].get();
    for (size_t ret_i = 0; ret_i < ret_stmt->value_.size(); ++ret_i) {
      auto ret_var = AsVarLike(ret_stmt->value_[ret_i]);
      if (!ret_var) continue;
      if (ret_var.get() == out_param) return ret_i;
    }
    return std::nullopt;
  }

  static std::optional<int64_t> GetConstIntValue(const ExprPtr& expr) {
    auto ci = As<ConstInt>(expr);
    if (!ci) return std::nullopt;
    return ci->value_;
  }

  static std::optional<int64_t> GetStaticTripCount(const ForStmtPtr& loop) {
    if (!loop) return std::nullopt;
    auto start = GetConstIntValue(loop->start_);
    auto stop = GetConstIntValue(loop->stop_);
    auto step = GetConstIntValue(loop->step_);
    if (!start.has_value() || !stop.has_value() || !step.has_value() || *step == 0) return std::nullopt;
    if ((*step > 0 && *stop <= *start) || (*step < 0 && *stop >= *start)) return int64_t{0};
    int64_t distance = *stop - *start;
    int64_t step_abs = *step > 0 ? *step : -*step;
    int64_t distance_abs = distance > 0 ? distance : -distance;
    return (distance_abs + step_abs - 1) / step_abs;
  }

  static std::optional<int64_t> GetKnownPositiveTripCount(const ForStmtPtr& loop) {
    auto static_trip_count = GetStaticTripCount(loop);
    if (static_trip_count.has_value()) return static_trip_count;
    if (!loop) return std::nullopt;
    auto step = GetConstIntValue(loop->step_);
    if (!step.has_value() || *step == 0) return std::nullopt;

    auto distance_expr = *step > 0 ? MakeSub(loop->stop_, loop->start_, loop->span_)
                                   : MakeSub(loop->start_, loop->stop_, loop->span_);
    distance_expr = arith::Analyzer().Simplify(distance_expr);
    auto distance = As<ConstInt>(distance_expr);
    if (!distance) return std::nullopt;
    if (distance->value_ <= 0) return int64_t{0};
    int64_t step_abs = *step > 0 ? *step : -*step;
    return (distance->value_ + step_abs - 1) / step_abs;
  }

  static std::optional<ExprPtr> SimplifyWithLoopBound(const ExprPtr& expr, const VarPtr& loop_var,
                                                      int64_t value) {
    if (!expr) return std::nullopt;
    arith::Analyzer analyzer;
    analyzer.Bind(loop_var, value, value + 1);
    return analyzer.Simplify(expr);
  }

  static std::optional<ExprPtr> SimplifyWithLoopValue(const ExprPtr& expr, const VarPtr& loop_var,
                                                      const ExprPtr& value) {
    if (!expr || !value) return std::nullopt;
    arith::Analyzer analyzer;
    analyzer.Bind(loop_var, value);
    return analyzer.Simplify(expr);
  }

  static std::optional<ExprPtr> GetLoopValueAtTrip(const ForStmtPtr& loop, int64_t trip_index) {
    if (!loop || trip_index < 0) return std::nullopt;
    auto step = GetConstIntValue(loop->step_);
    if (!step.has_value()) return std::nullopt;
    int64_t delta = trip_index * *step;
    if (delta == 0) return loop->start_;
    auto delta_expr = std::make_shared<ConstInt>(delta, DataType::INDEX, loop->span_);
    return arith::Analyzer().Simplify(MakeAdd(loop->start_, delta_expr, loop->span_));
  }

  static std::optional<OrderedLoopOffsets> GetOrderedLoopOffsets(const ExprPtr& expr, const ForStmtPtr& loop,
                                                                 const ExprPtr& first_loop_value,
                                                                 const ExprPtr& last_loop_value) {
    if (!expr || !loop || !first_loop_value || !last_loop_value) return std::nullopt;
    auto first_offset = SimplifyWithLoopValue(expr, loop->loop_var_, first_loop_value);
    auto last_offset = SimplifyWithLoopValue(expr, loop->loop_var_, last_loop_value);
    if (!first_offset.has_value() || !last_offset.has_value()) return std::nullopt;

    auto affine = ParseAffineInLoop(expr, loop->loop_var_.get());
    auto loop_step = GetConstIntValue(loop->step_);
    if (!affine.has_value() || !loop_step.has_value()) return std::nullopt;
    if (affine->coeff * *loop_step >= 0) {
      return OrderedLoopOffsets{*first_offset, *last_offset};
    }
    return OrderedLoopOffsets{*last_offset, *first_offset};
  }

  static std::optional<ExprPtr> ExpandLoopLocalExpr(
      const ExprPtr& expr, const std::unordered_map<const Var*, ExprPtr>& scalar_defs) {
    if (!expr) return std::nullopt;
    return transform_utils::Substitute(expr, scalar_defs);
  }

  static std::optional<FinalStoreInfo> AnalyzeFinalStore(const FunctionPtr& func, size_t out_param_index) {
    if (!func || out_param_index >= func->params_.size()) return std::nullopt;

    auto body_stmts = FlattenToStmts(func->body_);
    std::unordered_map<const Var*, AssignStmtPtr> var_defs;
    for (const auto& stmt : body_stmts) {
      if (auto assign = As<AssignStmt>(stmt)) var_defs[assign->var_.get()] = assign;
    }

    ReturnStmtPtr ret_stmt;
    for (const auto& stmt : body_stmts) {
      if (auto ret = As<ReturnStmt>(stmt)) {
        ret_stmt = ret;
        break;
      }
    }
    if (!ret_stmt) return std::nullopt;

    size_t total_out_refs = CountVarRefsInStmt(func->body_, func->params_[out_param_index].get());
    std::optional<FinalStoreInfo> result;
    size_t matched_refs = 0;
    for (size_t ret_i = 0; ret_i < ret_stmt->value_.size(); ++ret_i) {
      auto ret_var = AsVarLike(ret_stmt->value_[ret_i]);
      if (!ret_var) continue;
      auto def_it = var_defs.find(ret_var.get());
      if (def_it == var_defs.end()) continue;
      auto store_call = As<Call>(def_it->second->value_);
      if (!store_call || store_call->op_->name_ != "tile.store" || store_call->args_.size() < 3) continue;

      auto out_target = AsVarLike(store_call->args_[2]);
      if (!out_target || out_target.get() != func->params_[out_param_index].get()) continue;
      auto offset_tuple = As<MakeTuple>(store_call->args_[1]);
      auto tile_type = As<TileType>(store_call->args_[0]->GetType());
      if (!offset_tuple || !tile_type) return std::nullopt;

      matched_refs = CountVarRefsInStmt(def_it->second, func->params_[out_param_index].get());
      if (total_out_refs != matched_refs) return std::nullopt;

      result = FinalStoreInfo{ret_i, tile_type->shape_, offset_tuple->elements_};
      break;
    }
    return result;
  }

  static std::optional<CalleeRewriteAnalysis> AnalyzeAggregateWindowLoop(
      const FunctionPtr& func, const std::vector<size_t>& out_indices) {
    if (!func || out_indices.empty()) return std::nullopt;

    auto body_stmts = FlattenToStmts(func->body_);
    if (body_stmts.empty()) return std::nullopt;

    ReturnStmtPtr ret_stmt = As<ReturnStmt>(body_stmts.back());
    if (!ret_stmt) return std::nullopt;

    struct AggregateLoopOutputMatch {
      size_t out_param_index;
      size_t return_index;
      size_t iter_arg_index;
    };

    ForStmtPtr loop;
    std::vector<AggregateLoopOutputMatch> loop_matches;
    for (const auto& stmt : body_stmts) {
      auto candidate = As<ForStmt>(stmt);
      if (!candidate || candidate->iter_args_.empty()) continue;
      std::vector<AggregateLoopOutputMatch> candidate_matches;
      std::unordered_set<size_t> matched_iter_arg_indices;
      bool matches_all_outputs = true;

      for (const auto& out_param_index : out_indices) {
        std::optional<size_t> direct_return_index = FindReturnIndexForOutParam(func, out_param_index);
        VarPtr direct_returned;
        if (direct_return_index.has_value() && *direct_return_index < ret_stmt->value_.size()) {
          direct_returned = AsVarLike(ret_stmt->value_[*direct_return_index]);
        }

        bool matched_output = false;
        for (size_t i = 0; i < candidate->iter_args_.size() && i < candidate->return_vars_.size(); ++i) {
          auto init_var = AsVarLike(candidate->iter_args_[i]->initValue_);
          if (!init_var || init_var.get() != func->params_[out_param_index].get()) continue;

          std::optional<size_t> return_index = direct_return_index;
          if (direct_returned && direct_returned.get() != candidate->return_vars_[i].get()) {
            return_index = std::nullopt;
          }
          for (size_t ret_i = 0; ret_i < ret_stmt->value_.size(); ++ret_i) {
            if (return_index.has_value()) break;
            auto returned = AsVarLike(ret_stmt->value_[ret_i]);
            if (returned && returned.get() == candidate->return_vars_[i].get()) {
              return_index = ret_i;
              break;
            }
          }
          if (!return_index.has_value()) continue;

          if (!matched_iter_arg_indices.insert(i).second) return std::nullopt;
          candidate_matches.push_back(AggregateLoopOutputMatch{out_param_index, *return_index, i});
          matched_output = true;
          break;
        }
        if (!matched_output) {
          matches_all_outputs = false;
          break;
        }
      }

      if (!matches_all_outputs) continue;
      if (candidate->iter_args_.size() != candidate_matches.size() ||
          candidate->return_vars_.size() != candidate_matches.size()) {
        return std::nullopt;
      }

      if (loop) return std::nullopt;
      loop = candidate;
      loop_matches = std::move(candidate_matches);
    }
    if (!loop) return std::nullopt;
    if (loop_matches.size() != out_indices.size()) return std::nullopt;

    auto stop = GetConstIntValue(loop->stop_);
    auto step = GetConstIntValue(loop->step_);
    if (!stop.has_value() || !step.has_value()) {
      auto known_trip_count = GetKnownPositiveTripCount(loop);
      if (!known_trip_count.has_value() || *known_trip_count <= 0) return std::nullopt;
    } else if (*step <= 0) {
      return std::nullopt;
    }
    auto trip_count = GetKnownPositiveTripCount(loop);
    if (!trip_count.has_value() || *trip_count <= 0) return std::nullopt;
    auto first_loop_value = GetLoopValueAtTrip(loop, 0);
    auto last_loop_value = GetLoopValueAtTrip(loop, *trip_count - 1);
    if (!first_loop_value.has_value() || !last_loop_value.has_value()) return std::nullopt;

    auto loop_body_stmts = FlattenToStmts(loop->body_);
    YieldStmtPtr yield_stmt;
    struct AggregateUpdate {
      AssignStmtPtr assign;
      std::vector<ExprPtr> window_shape;
      std::vector<ExprPtr> offsets;
    };

    std::unordered_map<size_t, AggregateUpdate> updates_by_iter_arg_index;
    std::unordered_map<const Var*, ExprPtr> scalar_defs;
    for (const auto& stmt : loop_body_stmts) {
      if (auto assign = As<AssignStmt>(stmt)) {
        auto call = As<Call>(assign->value_);
        if (call) {
          const Var* updated_iter_arg = nullptr;
          std::vector<ExprPtr> window_shape;
          std::vector<ExprPtr> offsets;
          if (call->op_->name_ == "tile.store" && call->args_.size() >= 3) {
            auto out_arg = AsVarLike(call->args_[2]);
            auto offset_tuple = As<MakeTuple>(call->args_[1]);
            auto tile_type = As<TileType>(call->args_[0]->GetType());
            if (out_arg && offset_tuple && tile_type) {
              updated_iter_arg = out_arg.get();
              window_shape = tile_type->shape_;
              offsets = offset_tuple->elements_;
            }
          } else if (call->op_->name_ == "tensor.assemble" && call->args_.size() >= 3) {
            auto parent_arg = AsVarLike(call->args_[0]);
            auto offset_tuple = As<MakeTuple>(call->args_[2]);
            auto source_type = As<TensorType>(call->args_[1]->GetType());
            if (parent_arg && offset_tuple && source_type) {
              updated_iter_arg = parent_arg.get();
              window_shape = source_type->shape_;
              offsets = offset_tuple->elements_;
            }
          }

          bool matched_update = false;
          if (updated_iter_arg) {
            for (const auto& match : loop_matches) {
              if (updated_iter_arg != loop->iter_args_[match.iter_arg_index].get()) continue;
              if (updates_by_iter_arg_index.count(match.iter_arg_index)) return std::nullopt;
              updates_by_iter_arg_index.emplace(
                  match.iter_arg_index, AggregateUpdate{assign, std::move(window_shape), std::move(offsets)});
              matched_update = true;
              break;
            }
          }
          if (matched_update) continue;
        }
        if (As<ScalarType>(assign->var_->GetType())) {
          scalar_defs[assign->var_.get()] = assign->value_;
        }
        continue;
      }
      if (auto yield = As<YieldStmt>(stmt)) {
        if (yield_stmt || yield->value_.size() != loop->return_vars_.size()) return std::nullopt;
        yield_stmt = yield;
      }
    }

    if (!yield_stmt || updates_by_iter_arg_index.size() != loop_matches.size()) return std::nullopt;

    std::unordered_set<const Var*> allowed;
    for (const auto& param : func->params_) allowed.insert(param.get());
    allowed.insert(loop->loop_var_.get());

    CalleeRewriteAnalysis analysis;
    analysis.kind = RewriteKind::AggregateWindowLoop;

    for (const auto& match : loop_matches) {
      auto update_it = updates_by_iter_arg_index.find(match.iter_arg_index);
      if (update_it == updates_by_iter_arg_index.end()) return std::nullopt;
      const auto& update = update_it->second;
      auto store_assign = update.assign;

      auto yielded = AsVarLike(yield_stmt->value_[match.iter_arg_index]);
      if (!yielded || yielded.get() != store_assign->var_.get()) return std::nullopt;

      if (!As<TensorType>(loop->iter_args_[match.iter_arg_index]->GetType()) ||
          !As<TensorType>(loop->return_vars_[match.iter_arg_index]->GetType())) {
        return std::nullopt;
      }

      size_t total_out_refs = CountVarRefsInStmt(func->body_, func->params_[match.out_param_index].get());
      size_t store_out_refs = CountVarRefsInStmt(store_assign, func->params_[match.out_param_index].get());
      if (total_out_refs != store_out_refs + 1) return std::nullopt;

      size_t total_iter_refs = CountVarRefsInStmt(loop->body_, loop->iter_args_[match.iter_arg_index].get());
      size_t store_iter_refs = CountVarRefsInStmt(store_assign, loop->iter_args_[match.iter_arg_index].get());
      if (total_iter_refs != store_iter_refs) return std::nullopt;

      auto out_tensor_type = As<TensorType>(func->params_[match.out_param_index]->GetType());
      if (!out_tensor_type) return std::nullopt;
      if (update.offsets.size() != update.window_shape.size() ||
          update.offsets.size() != out_tensor_type->shape_.size()) {
        return std::nullopt;
      }

      std::vector<ExprPtr> base_offsets;
      std::vector<ExprPtr> local_offsets;
      std::vector<ExprPtr> window_shape;
      for (size_t i = 0; i < update.offsets.size(); ++i) {
        auto expanded = ExpandLoopLocalExpr(update.offsets[i], scalar_defs);
        if (!expanded.has_value()) return std::nullopt;
        if (!ExprReferencesOnlyVarsIn(*expanded, allowed)) return std::nullopt;

        auto ordered_offsets = GetOrderedLoopOffsets(*expanded, loop, *first_loop_value, *last_loop_value);
        if (!ordered_offsets.has_value()) return std::nullopt;

        auto span_expr = arith::Analyzer().Simplify(
            MakeAdd(MakeSub(ordered_offsets->max, ordered_offsets->min, func->span_), update.window_shape[i],
                    func->span_));
        auto span_ci = As<ConstInt>(span_expr);
        if (!span_ci || span_ci->value_ <= 0) return std::nullopt;

        base_offsets.push_back(ordered_offsets->min);
        local_offsets.push_back(arith::Analyzer().Simplify(
            MakeSub(update.offsets[i], ordered_offsets->min, update.offsets[i]->span_)));
        window_shape.push_back(std::make_shared<ConstInt>(span_ci->value_, DataType::INDEX, func->span_));
      }

      if (AreExprVectorsEqual(window_shape, out_tensor_type->shape_) && IsAllZeroOffsets(base_offsets)) {
        return std::nullopt;
      }

      analysis.outputs.push_back(OutputRewriteInfo{
          match.out_param_index, match.return_index, out_tensor_type->shape_, std::move(window_shape),
          std::move(base_offsets), std::move(local_offsets), match.iter_arg_index});
    }

    return analysis;
  }

  AnalysisMap Analyze(const ProgramPtr& program) {
    AnalysisMap analyses;
    for (const auto& [gvar, func] : program->functions_) {
      if (!func || pypto::codegen::IsBuiltinOp(func->name_) || func->func_type_ != FunctionType::InCore) {
        continue;
      }
      if (ContainsGeneratedChunkLoop(func)) continue;

      auto out_indices = CollectOutParamIndices(func);
      if (out_indices.empty()) continue;

      CalleeRewriteAnalysis analysis;
      bool all_final = true;
      for (const auto& out_index : out_indices) {
        auto info = AnalyzeFinalStore(func, out_index);
        if (!info.has_value()) {
          all_final = false;
          break;
        }

        auto out_tensor_type = As<TensorType>(func->params_[out_index]->GetType());
        if (!out_tensor_type) {
          all_final = false;
          break;
        }
        if (AreExprVectorsEqual(info->window_shape, out_tensor_type->shape_) &&
            IsAllZeroOffsets(info->offsets)) {
          all_final = false;
          break;
        }

        std::unordered_set<const Var*> allowed_params;
        for (const auto& param : func->params_) allowed_params.insert(param.get());
        bool exprs_ok = true;
        for (const auto& expr : info->window_shape) {
          if (!ExprReferencesOnlyVarsIn(expr, allowed_params)) {
            exprs_ok = false;
            break;
          }
        }
        for (const auto& expr : info->offsets) {
          if (!ExprReferencesOnlyVarsIn(expr, allowed_params)) {
            exprs_ok = false;
            break;
          }
        }
        if (!exprs_ok) {
          all_final = false;
          break;
        }

        std::vector<ExprPtr> local_zero_offsets;
        local_zero_offsets.reserve(info->offsets.size());
        for (size_t i = 0; i < info->offsets.size(); ++i) {
          local_zero_offsets.push_back(std::make_shared<ConstInt>(0, DataType::INDEX, func->span_));
        }
        analysis.outputs.push_back(OutputRewriteInfo{out_index, info->return_index, out_tensor_type->shape_,
                                                     info->window_shape, info->offsets, local_zero_offsets,
                                                     SIZE_MAX});
      }
      if (all_final && !analysis.outputs.empty()) {
        analysis.kind = RewriteKind::FinalStore;
        analyses.emplace(func->name_, std::move(analysis));
        continue;
      }

      auto aggregate_analysis = AnalyzeAggregateWindowLoop(func, out_indices);
      if (aggregate_analysis.has_value() && !aggregate_analysis->outputs.empty()) {
        analyses.emplace(func->name_, std::move(*aggregate_analysis));
      }
    }
    return analyses;
  }

  FunctionPtr RewriteCallee(const ProgramPtr& program, const FunctionPtr& func,
                            const CalleeRewriteAnalysis& analysis) {
    if (!func) return nullptr;

    std::vector<VarPtr> new_params;
    new_params.reserve(func->params_.size());
    std::vector<TypePtr> new_return_types = func->return_types_;
    auto new_param_directions = func->param_directions_;

    std::unordered_map<const Var*, ExprPtr> seed;
    for (size_t i = 0; i < func->params_.size(); ++i) {
      auto param_type = func->params_[i]->GetType();
      auto rewrite_it =
          std::find_if(analysis.outputs.begin(), analysis.outputs.end(),
                       [i](const OutputRewriteInfo& info) { return info.out_param_index == i; });
      if (rewrite_it != analysis.outputs.end()) {
        auto out_tensor_type = As<TensorType>(func->params_[i]->GetType());
        if (!out_tensor_type) return nullptr;

        std::optional<TensorView> new_view = std::nullopt;
        if (out_tensor_type->tensor_view_.has_value()) {
          new_view = out_tensor_type->tensor_view_;
          if (new_view->stride.empty()) {
            if (new_view->layout == TensorLayout::NZ) return nullptr;
            new_view->stride = tensor_view_semantics::BuildLogicalStridesFromLayout(out_tensor_type->shape_,
                                                                                    new_view->layout);
          }
          if (!new_view->valid_shape.empty()) new_view->valid_shape = rewrite_it->window_shape;
        } else {
          auto parent_strides = ComputeRowMajorStrides(rewrite_it->parent_shape);
          if (parent_strides.empty() || parent_strides.size() != rewrite_it->window_shape.size()) {
            return nullptr;
          }
          new_view = TensorView(std::move(parent_strides), TensorLayout::ND);
        }

        param_type = std::make_shared<TensorType>(rewrite_it->window_shape, out_tensor_type->dtype_,
                                                  out_tensor_type->memref_, new_view);
        new_return_types[rewrite_it->return_index] = param_type;
      }

      auto new_param =
          std::make_shared<Var>(func->params_[i]->name_hint_, param_type, func->params_[i]->span_);
      new_params.push_back(new_param);
      seed[func->params_[i].get()] = new_param;
    }

    auto cloned_name = MakeUniqueFunctionName(program, func->name_ + "__windowed");
    auto cloned = DeepClone(func->body_, seed);
    std::unordered_map<const Var*, ExprPtr> body_subst = seed;
    for (const auto& [old_var, new_var] : cloned.var_map) {
      body_subst[old_var] = new_var;
    }

    std::vector<OutputRewriteInfo> localized_outputs = analysis.outputs;
    for (auto& output : localized_outputs) {
      for (auto& offset : output.callsite_offsets) {
        offset = transform_utils::Substitute(offset, body_subst);
      }
      for (auto& offset : output.local_store_offsets) {
        offset = transform_utils::Substitute(offset, body_subst);
      }
    }
    StmtPtr new_body = cloned.cloned_body;

    if (analysis.kind == RewriteKind::AggregateWindowLoop) {
      auto find_aggregate_loop = [&](const StmtPtr& body) -> ForStmtPtr {
        auto body_stmts = FlattenToStmts(body);
        auto ret_stmt = body_stmts.empty() ? nullptr : As<ReturnStmt>(body_stmts.back());
        if (!ret_stmt) return nullptr;

        ForStmtPtr matched_loop;
        for (const auto& stmt : body_stmts) {
          auto candidate = As<ForStmt>(stmt);
          if (!candidate) continue;

          bool matches_outputs = true;
          for (const auto& output : analysis.outputs) {
            if (output.iter_arg_index >= candidate->iter_args_.size() ||
                output.iter_arg_index >= candidate->return_vars_.size() ||
                output.return_index >= ret_stmt->value_.size()) {
              matches_outputs = false;
              break;
            }
            auto init_var = AsVarLike(candidate->iter_args_[output.iter_arg_index]->initValue_);
            auto returned = AsVarLike(ret_stmt->value_[output.return_index]);
            if (!init_var || !returned) {
              matches_outputs = false;
              break;
            }
            if (init_var.get() != new_params[output.out_param_index].get() ||
                returned.get() != candidate->return_vars_[output.iter_arg_index].get()) {
              matches_outputs = false;
              break;
            }
          }
          if (!matches_outputs) continue;
          if (matched_loop) return nullptr;
          matched_loop = candidate;
        }
        return matched_loop;
      };

      auto cloned_loop = find_aggregate_loop(new_body);
      if (!cloned_loop) return nullptr;

      std::unordered_map<const Var*, TypePtr> narrowed_return_vars;
      for (const auto& output : analysis.outputs) {
        if (output.iter_arg_index >= cloned_loop->return_vars_.size()) return nullptr;
        narrowed_return_vars.emplace(cloned_loop->return_vars_[output.iter_arg_index].get(),
                                     new_return_types[output.return_index]);
      }

      class AggregateLoopTypeLocalizer : public IRMutator {
       public:
        explicit AggregateLoopTypeLocalizer(
            const std::unordered_map<const Var*, TypePtr>& narrowed_return_vars)
            : narrowed_return_vars_(narrowed_return_vars) {}

       protected:
        StmtPtr VisitStmt_(const ForStmtPtr& op) override {
          std::vector<const Var*> old_iter_args_to_erase;
          bool changed = false;
          for (size_t i = 0; i < op->return_vars_.size() && i < op->iter_args_.size(); ++i) {
            auto it = narrowed_return_vars_.find(op->return_vars_[i].get());
            if (it == narrowed_return_vars_.end()) continue;
            auto old_iter = op->iter_args_[i];
            auto old_ret = op->return_vars_[i];
            auto new_iter = std::make_shared<IterArg>(old_iter->name_hint_, it->second, old_iter->initValue_,
                                                      old_iter->span_);
            auto new_ret = std::make_shared<Var>(old_ret->name_hint_, it->second, old_ret->span_);
            var_remap_[old_iter.get()] = new_iter;
            var_remap_[old_ret.get()] = new_ret;
            old_iter_args_to_erase.push_back(old_iter.get());
            changed = true;
          }
          auto new_stmt = IRMutator::VisitStmt_(op);
          for (const auto* old_iter : old_iter_args_to_erase) {
            var_remap_.erase(old_iter);
          }
          return changed ? new_stmt : op;
        }

       private:
        const std::unordered_map<const Var*, TypePtr>& narrowed_return_vars_;
      };

      AggregateLoopTypeLocalizer type_localizer(narrowed_return_vars);
      new_body = type_localizer.VisitStmt(new_body);

      auto typed_loop = find_aggregate_loop(new_body);
      if (!typed_loop) return nullptr;

      std::unordered_map<const Var*, OutputRewriteInfo> out_info_by_var;
      std::unordered_map<const Var*, TypePtr> new_store_types;
      std::unordered_map<const Var*, ExprPtr> new_out_vars;
      for (const auto& output : localized_outputs) {
        if (output.iter_arg_index >= typed_loop->iter_args_.size()) return nullptr;
        auto iter_arg = typed_loop->iter_args_[output.iter_arg_index];
        out_info_by_var.emplace(iter_arg.get(), output);
        new_out_vars.emplace(iter_arg.get(), iter_arg);
        new_store_types.emplace(iter_arg.get(), new_return_types[output.return_index]);
      }

      WindowWriteLocalizer localizer(out_info_by_var, new_out_vars, new_store_types);
      new_body = localizer.VisitStmt(new_body);
    } else {
      std::unordered_map<const Var*, OutputRewriteInfo> out_info_by_var;
      std::unordered_map<const Var*, TypePtr> new_store_types;
      std::unordered_map<const Var*, ExprPtr> new_out_vars;
      for (const auto& output : localized_outputs) {
        auto new_out = new_params[output.out_param_index];
        out_info_by_var.emplace(new_out.get(), output);
        new_store_types.emplace(new_out.get(), new_out->GetType());
        new_out_vars.emplace(new_out.get(), new_out);
      }
      WindowWriteLocalizer localizer(out_info_by_var, new_out_vars, new_store_types);
      new_body = localizer.VisitStmt(new_body);
    }

    return std::make_shared<Function>(cloned_name, new_params, new_param_directions, new_return_types,
                                      new_body, func->span_, func->func_type_, func->level_, func->role_,
                                      func->attrs_);
  }
};

}  // namespace

// ============================================================================
// Pass entry point
// ============================================================================

namespace pass {

Pass OptimizeOrchTensors() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    // Collect InCore function names
    std::unordered_set<std::string> incore_names;
    for (const auto& [gvar, func] : program->functions_) {
      if (func->func_type_ == FunctionType::InCore) {
        incore_names.insert(func->name_);
      }
    }
    if (incore_names.empty()) return program;

    // Pattern 1: Iter-arg reuse (may remove Out params)
    auto p1 = IterArgReuseOptimizer().Run(program, incore_names);

    // Pattern 2: Assemble parent strides (sees Pattern 1 results)
    auto p2 = AssembleParentStridesOptimizer().Run(p1, incore_names);

    // Pattern 3: Assemble-loop rewrite (sees Pattern 2 results)
    auto p3 = AssembleLoopRewriter().Run(p2, incore_names);

    // Pattern 4: Slice input strides (propagate parent strides to In params)
    auto p4 = SliceInputStridesOptimizer().Run(p3, incore_names);

    // Pattern 5: Static output windows with precomputed runtime Tensor descriptors
    return StaticOutputWindowPlanner().Run(p4);
  };

  return CreateProgramPass(pass_func, "OptimizeOrchTensors", kOptimizeOrchTensorsProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
