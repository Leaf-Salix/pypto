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

#include <any>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/codegen/orchestration/orchestration_analysis.h"
#include "pypto/codegen/orchestration/template_registry.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"

namespace pypto {
namespace ir {

namespace {

constexpr const char* kStableRegionTemplateKey = "stable_region_template_key";
constexpr const char* kManualTaskIndex = "manual_task_index";
constexpr const char* kManualDepIndices = "manual_dep_indices";

using ::pypto::codegen::IsBuiltinOp;
using ::pypto::codegen::IsTensorOp;
using ::pypto::codegen::orchestration::AllowedIfFlagWindow;
using ::pypto::codegen::orchestration::GetStableRegionTemplates;
using ::pypto::codegen::orchestration::KernelNameMatchesToken;
using ::pypto::codegen::orchestration::StableRegionKind;
using ::pypto::codegen::orchestration::StableRegionTemplate;

struct RegionCallAttrs {
  std::string template_key;
  int task_index = 0;
  std::vector<int> dep_indices;
};

CallPtr GetDirectCall(const StmtPtr& stmt) {
  if (auto assign = As<AssignStmt>(stmt)) {
    return As<Call>(assign->value_);
  }
  if (auto eval = As<EvalStmt>(stmt)) {
    return As<Call>(eval->expr_);
  }
  return nullptr;
}

bool IsTaskCallStmt(const StmtPtr& stmt) {
  auto call = GetDirectCall(stmt);
  return call && !IsBuiltinOp(call->op_->name_);
}

bool ArgDirectionReads(ArgDirection dir) {
  return dir == ArgDirection::Input || dir == ArgDirection::InOut;
}

bool ArgDirectionWrites(ArgDirection dir) {
  return dir == ArgDirection::Output || dir == ArgDirection::InOut ||
         dir == ArgDirection::OutputExisting;
}

ArgDirection GetArgDirectionOrDefault(const CallPtr& call, size_t arg_index) {
  auto directions = call->GetArgDirections();
  if (directions.empty()) {
    return ArgDirection::Input;
  }
  INTERNAL_CHECK_SPAN(directions.size() == call->args_.size(), call->span_)
      << "Call arg_directions size (" << directions.size() << ") must match args size ("
      << call->args_.size() << ")";
  return directions[arg_index];
}

bool ArgDirectionsMatchPattern(const CallPtr& call, const std::vector<ArgDirection>& expected) {
  if (expected.empty()) {
    return true;
  }
  auto actual = call->GetArgDirections();
  if (actual.size() != expected.size()) {
    return false;
  }
  return actual == expected;
}

bool TemplateStageMatchesCall(const StableRegionTemplate& templ, size_t stage_index, const CallPtr& call) {
  if (!KernelNameMatchesToken(call->op_->name_, templ.kernel_name_tokens[stage_index])) {
    return false;
  }
  if (templ.arg_direction_patterns.empty()) {
    return true;
  }
  INTERNAL_CHECK(templ.arg_direction_patterns.size() == templ.kernel_name_tokens.size())
      << "Internal error: stable-region template '" << templ.template_key
      << "' arg_direction_patterns size must match kernel_name_tokens size";
  return ArgDirectionsMatchPattern(call, templ.arg_direction_patterns[stage_index]);
}

std::vector<const Var*> CollectExprVarUses(const ExprPtr& expr) {
  class Collector : public IRVisitor {
   public:
    std::vector<const Var*> vars;

   protected:
    void VisitExpr_(const VarPtr& op) override { vars.push_back(op.get()); }
    void VisitExpr_(const IterArgPtr& op) override { vars.push_back(op.get()); }
    void VisitExpr_(const MemRefPtr& op) override { vars.push_back(op.get()); }
  };

  Collector collector;
  collector.VisitExpr(expr);
  return collector.vars;
}

bool IsAllowedBetweenTemplateTasks(const StmtPtr& stmt, const std::unordered_set<const Stmt*>& task_stmts) {
  if (task_stmts.count(stmt.get()) > 0) {
    return true;
  }
  if (As<ForStmt>(stmt) || As<WhileStmt>(stmt) || As<IfStmt>(stmt) || As<ScopeStmt>(stmt)) {
    return false;
  }
  auto call = GetDirectCall(stmt);
  if (!call) {
    return true;
  }
  if (!IsTensorOp(call->op_->name_)) {
    return false;
  }
  const std::string& op_name = call->op_->name_;
  return op_name == "tensor.create" || op_name == "tensor.slice";
}

bool HasUnsafeOpenTaskBoundary(const SeqStmtsPtr& seq, size_t start_stmt, size_t end_stmt,
                               const std::vector<size_t>& matched_task_stmt_indices) {
  std::unordered_set<size_t> matched_task_indices;
  matched_task_indices.reserve(matched_task_stmt_indices.size());
  for (size_t index : matched_task_stmt_indices) {
    matched_task_indices.insert(index);
  }

  std::unordered_map<const Var*, std::unordered_set<size_t>> task_origins_by_var;
  for (size_t i = 0; i < seq->stmts_.size(); ++i) {
    if (auto assign = As<AssignStmt>(seq->stmts_[i])) {
      auto call = As<Call>(assign->value_);
      if (call && !IsBuiltinOp(call->op_->name_)) {
        task_origins_by_var[assign->var_.get()].insert(i);
        for (size_t arg_index = 0; arg_index < call->args_.size(); ++arg_index) {
          if (!ArgDirectionWrites(GetArgDirectionOrDefault(call, arg_index))) {
            continue;
          }
          for (const Var* var : CollectExprVarUses(call->args_[arg_index])) {
            task_origins_by_var[var].insert(i);
          }
        }
        continue;
      }

      std::unordered_set<size_t> origins;
      for (const Var* var : CollectExprVarUses(assign->value_)) {
        auto origin_it = task_origins_by_var.find(var);
        if (origin_it == task_origins_by_var.end()) {
          continue;
        }
        origins.insert(origin_it->second.begin(), origin_it->second.end());
      }
      if (!origins.empty()) {
        task_origins_by_var[assign->var_.get()] = std::move(origins);
      }
    }
  }

  for (size_t index : matched_task_stmt_indices) {
    auto call = GetDirectCall(seq->stmts_[index]);
    INTERNAL_CHECK(call) << "Internal error: matched task statement has no direct call";
    for (size_t arg_index = 0; arg_index < call->args_.size(); ++arg_index) {
      if (!ArgDirectionReads(GetArgDirectionOrDefault(call, arg_index))) {
        continue;
      }
      for (const Var* var : CollectExprVarUses(call->args_[arg_index])) {
        auto origin_it = task_origins_by_var.find(var);
        if (origin_it == task_origins_by_var.end()) {
          continue;
        }
        for (size_t producer_index : origin_it->second) {
          if (producer_index < start_stmt || producer_index > end_stmt) {
            return true;
          }
        }
      }
    }
  }

  for (size_t i = 0; i < seq->stmts_.size(); ++i) {
    if (i >= start_stmt && i <= end_stmt) {
      continue;
    }
    auto call = GetDirectCall(seq->stmts_[i]);
    if (!call || IsBuiltinOp(call->op_->name_)) {
      continue;
    }
    for (size_t arg_index = 0; arg_index < call->args_.size(); ++arg_index) {
      if (!ArgDirectionReads(GetArgDirectionOrDefault(call, arg_index))) {
        continue;
      }
      for (const Var* var : CollectExprVarUses(call->args_[arg_index])) {
        auto origin_it = task_origins_by_var.find(var);
        if (origin_it == task_origins_by_var.end()) {
          continue;
        }
        for (size_t producer_index : origin_it->second) {
          if (matched_task_indices.count(producer_index) > 0) {
            return true;
          }
        }
      }
    }
  }

  return false;
}

std::vector<std::pair<std::string, std::any>> UpsertAttr(
    std::vector<std::pair<std::string, std::any>> attrs, const std::string& key, std::any value) {
  for (auto& [k, v] : attrs) {
    if (k == key) {
      v = std::move(value);
      return attrs;
    }
  }
  attrs.emplace_back(key, std::move(value));
  return attrs;
}

std::vector<std::pair<std::string, std::any>> WithStableRegionAttrs(
    std::vector<std::pair<std::string, std::any>> attrs, const RegionCallAttrs& region_attrs) {
  attrs = UpsertAttr(std::move(attrs), kStableRegionTemplateKey, region_attrs.template_key);
  attrs = UpsertAttr(std::move(attrs), kManualTaskIndex, region_attrs.task_index);
  attrs = UpsertAttr(std::move(attrs), kManualDepIndices, region_attrs.dep_indices);
  return attrs;
}

CallPtr CloneCallWithStableRegionAttrs(const CallPtr& call, const RegionCallAttrs& region_attrs) {
  auto attrs = WithStableRegionAttrs(call->attrs_, region_attrs);
  return std::make_shared<Call>(call->op_, call->args_, call->kwargs_, std::move(attrs), call->GetType(),
                                call->span_);
}

StmtPtr ReplaceDirectCall(const StmtPtr& stmt, const CallPtr& new_call) {
  if (auto assign = As<AssignStmt>(stmt)) {
    auto replacement = MutableCopy(assign);
    replacement->value_ = new_call;
    return replacement;
  }
  if (auto eval = As<EvalStmt>(stmt)) {
    auto replacement = MutableCopy(eval);
    replacement->expr_ = new_call;
    return replacement;
  }
  INTERNAL_CHECK(false) << "Internal error: direct call stmt is neither AssignStmt nor EvalStmt";
  return stmt;
}

bool ExprIsScalar(const ExprPtr& expr) { return expr && As<ScalarType>(expr->GetType()) != nullptr; }

bool IsScalarOnlyBranchPrefixStmt(const StmtPtr& stmt) {
  if (As<ForStmt>(stmt) || As<WhileStmt>(stmt) || As<IfStmt>(stmt) || As<ScopeStmt>(stmt) || As<ReturnStmt>(stmt) ||
      As<YieldStmt>(stmt)) {
    return false;
  }

  if (auto call = GetDirectCall(stmt)) {
    return IsBuiltinOp(call->op_->name_) && ExprIsScalar(call);
  }

  if (auto assign = As<AssignStmt>(stmt)) {
    return As<ScalarType>(assign->var_->GetType()) != nullptr && ExprIsScalar(assign->value_);
  }

  if (auto eval = As<EvalStmt>(stmt)) {
    return ExprIsScalar(eval->expr_);
  }

  return true;
}

bool BranchEndsWithScalarYield(const StmtPtr& stmt) {
  auto yield = As<YieldStmt>(stmt);
  if (yield) {
    return yield->value_.size() == 1 && ExprIsScalar(yield->value_[0]);
  }

  auto seq = As<SeqStmts>(stmt);
  if (!seq || seq->stmts_.empty()) {
    return false;
  }
  for (size_t i = 0; i + 1 < seq->stmts_.size(); ++i) {
    if (!IsScalarOnlyBranchPrefixStmt(seq->stmts_[i])) {
      return false;
    }
  }
  auto trailing_yield = As<YieldStmt>(seq->stmts_.back());
  return trailing_yield && trailing_yield->value_.size() == 1 && ExprIsScalar(trailing_yield->value_[0]);
}

bool IsSupportedFlagIfStmt(const IfStmtPtr& if_stmt) {
  if (!if_stmt || !if_stmt->else_body_.has_value() || if_stmt->return_vars_.size() != 1) {
    return false;
  }
  if (!As<ScalarType>(if_stmt->return_vars_[0]->GetType())) {
    return false;
  }
  return BranchEndsWithScalarYield(if_stmt->then_body_) && BranchEndsWithScalarYield(*if_stmt->else_body_);
}

bool ContainsNestedLoop(const StmtPtr& stmt) {
  class NestedLoopFinder : public IRVisitor {
   public:
    bool has_nested_loop = false;

   protected:
    void VisitStmt_(const ForStmtPtr&) override { has_nested_loop = true; }
    void VisitStmt_(const WhileStmtPtr&) override { has_nested_loop = true; }
  };

  NestedLoopFinder finder;
  finder.VisitStmt(stmt);
  return finder.has_nested_loop;
}

const AllowedIfFlagWindow* FindAllowedIfFlagWindow(const StableRegionTemplate& templ, size_t after_task_index,
                                                   size_t before_task_index) {
  for (const auto& window : templ.allowed_if_flag_windows) {
    if (window.after_task_index == after_task_index && window.before_task_index == before_task_index) {
      return &window;
    }
  }
  return nullptr;
}

class StableRegionIdentifier : public IRMutator {
 protected:
  StmtPtr VisitStmt_(const ForStmtPtr& for_stmt) override {
    if (auto body_seq = As<SeqStmts>(for_stmt->body_)) {
      auto marks = MatchLoopBodyStableRegion(for_stmt, body_seq);
      if (!marks.empty()) {
        auto new_body = RewriteSeqWithMarks(body_seq, marks);
        if (new_body.get() != for_stmt->body_.get()) {
          auto replacement = MutableCopy(for_stmt);
          replacement->body_ = new_body;
          return replacement;
        }
      }
    }
    return IRMutator::VisitStmt_(for_stmt);
  }

  StmtPtr VisitStmt_(const SeqStmtsPtr& seq) override {
    std::unordered_map<const Call*, RegionCallAttrs> marks = MatchStableRegions(seq);
    if (marks.empty()) {
      return IRMutator::VisitStmt_(seq);
    }
    return RewriteSeqWithMarks(seq, marks);
  }

 private:
  StmtPtr RewriteSeqWithMarks(const SeqStmtsPtr& seq, const std::unordered_map<const Call*, RegionCallAttrs>& marks) {
    std::vector<StmtPtr> new_stmts;
    bool changed = false;
    new_stmts.reserve(seq->stmts_.size());

    for (const auto& stmt : seq->stmts_) {
      auto call = GetDirectCall(stmt);
      auto mark_it = call ? marks.find(call.get()) : marks.end();
      if (call && mark_it != marks.end()) {
        auto new_call = CloneCallWithStableRegionAttrs(call, mark_it->second);
        new_stmts.push_back(ReplaceDirectCall(stmt, new_call));
        changed = true;
        continue;
      }

      auto new_stmt = IRMutator::VisitStmt(stmt);
      if (new_stmt.get() != stmt.get()) {
        changed = true;
      }
      new_stmts.push_back(std::move(new_stmt));
    }

    if (changed) {
      return SeqStmts::Flatten(std::move(new_stmts), seq->span_);
    }
    return seq;
  }

  std::unordered_map<const Call*, RegionCallAttrs> MatchStableRegions(const SeqStmtsPtr& seq) {
    std::unordered_map<const Call*, RegionCallAttrs> result;
    std::vector<size_t> task_stmt_indices;
    for (size_t i = 0; i < seq->stmts_.size(); ++i) {
      if (IsTaskCallStmt(seq->stmts_[i])) {
        task_stmt_indices.push_back(i);
      }
    }

    size_t task_pos = 0;
    while (task_pos < task_stmt_indices.size()) {
      bool matched = false;
      for (const auto& templ : GetStableRegionTemplates()) {
        if (templ.region_kind != StableRegionKind::StraightLine) {
          continue;
        }
        const auto& tokens = templ.kernel_name_tokens;
        if (task_pos + tokens.size() > task_stmt_indices.size()) {
          continue;
        }

        bool stage_match = true;
        std::unordered_set<const Stmt*> task_stmts;
        for (size_t j = 0; j < tokens.size(); ++j) {
          const auto& stmt = seq->stmts_[task_stmt_indices[task_pos + j]];
          auto call = GetDirectCall(stmt);
          INTERNAL_CHECK(call) << "Internal error: task index without direct call";
          if (!TemplateStageMatchesCall(templ, j, call)) {
            stage_match = false;
            break;
          }
          task_stmts.insert(stmt.get());
        }
        if (!stage_match) {
          continue;
        }

        size_t start_stmt = task_stmt_indices[task_pos];
        size_t end_stmt = task_stmt_indices[task_pos + tokens.size() - 1];
        std::vector<size_t> matched_task_stmt_indices;
        matched_task_stmt_indices.reserve(tokens.size());
        for (size_t j = 0; j < tokens.size(); ++j) {
          matched_task_stmt_indices.push_back(task_stmt_indices[task_pos + j]);
        }

        bool safe_window = true;
        for (size_t i = start_stmt; i <= end_stmt; ++i) {
          if (!IsAllowedBetweenTemplateTasks(seq->stmts_[i], task_stmts)) {
            safe_window = false;
            break;
          }
        }
        if (safe_window && HasUnsafeOpenTaskBoundary(seq, start_stmt, end_stmt, matched_task_stmt_indices)) {
          safe_window = false;
        }
        if (!safe_window) {
          continue;
        }

        for (size_t j = 0; j < tokens.size(); ++j) {
          auto call = GetDirectCall(seq->stmts_[task_stmt_indices[task_pos + j]]);
          RegionCallAttrs attrs;
          attrs.template_key = templ.template_key;
          attrs.task_index = static_cast<int>(j);
          if (j > 0) {
            attrs.dep_indices.push_back(static_cast<int>(j - 1));
          }
          result[call.get()] = std::move(attrs);
        }
        task_pos += tokens.size();
        matched = true;
        break;
      }
      if (!matched) {
        ++task_pos;
      }
    }
    return result;
  }

  std::unordered_map<const Call*, RegionCallAttrs> MatchLoopBodyStableRegion(const ForStmtPtr& for_stmt,
                                                                              const SeqStmtsPtr& body_seq) {
    if (ContainsNestedLoop(body_seq)) {
      return {};
    }

    for (const auto& templ : GetStableRegionTemplates()) {
      if (templ.region_kind != StableRegionKind::LoopBody) {
        continue;
      }
      auto marks = TryMatchLoopBodyTemplate(for_stmt, body_seq, templ);
      if (!marks.empty()) {
        return marks;
      }
    }
    return {};
  }

  std::unordered_map<const Call*, RegionCallAttrs> TryMatchLoopBodyTemplate(const ForStmtPtr& for_stmt,
                                                                             const SeqStmtsPtr& body_seq,
                                                                             const StableRegionTemplate& templ) {
    if (templ.loop_iter_arg_count != 0 && for_stmt->iter_args_.size() != templ.loop_iter_arg_count) {
      return {};
    }
    if (templ.loop_return_var_count != 0 && for_stmt->return_vars_.size() != templ.loop_return_var_count) {
      return {};
    }

    std::vector<size_t> matched_task_stmt_indices;
    matched_task_stmt_indices.reserve(templ.kernel_name_tokens.size());
    std::unordered_map<size_t, size_t> observed_if_flag_counts;
    std::unordered_set<const Stmt*> task_stmts;
    size_t next_task_index = 0;

    for (size_t stmt_index = 0; stmt_index < body_seq->stmts_.size(); ++stmt_index) {
      const auto& stmt = body_seq->stmts_[stmt_index];

      if (auto call = GetDirectCall(stmt)) {
        if (!IsBuiltinOp(call->op_->name_)) {
          if (next_task_index >= templ.kernel_name_tokens.size() ||
              !TemplateStageMatchesCall(templ, next_task_index, call)) {
            return {};
          }
          matched_task_stmt_indices.push_back(stmt_index);
          task_stmts.insert(stmt.get());
          ++next_task_index;
          continue;
        }
        if (!IsAllowedBetweenTemplateTasks(stmt, task_stmts)) {
          return {};
        }
        continue;
      }

      if (auto if_stmt = As<IfStmt>(stmt)) {
        if (next_task_index == 0 || next_task_index >= templ.kernel_name_tokens.size()) {
          return {};
        }
        const AllowedIfFlagWindow* window = FindAllowedIfFlagWindow(templ, next_task_index - 1, next_task_index);
        if (!window || !IsSupportedFlagIfStmt(if_stmt)) {
          return {};
        }
        observed_if_flag_counts[next_task_index]++;
        continue;
      }

      if ((As<ForStmt>(stmt) || As<WhileStmt>(stmt) || As<ScopeStmt>(stmt)) && !task_stmts.count(stmt.get())) {
        return {};
      }
    }

    if (matched_task_stmt_indices.size() != templ.kernel_name_tokens.size()) {
      return {};
    }

    for (const auto& window : templ.allowed_if_flag_windows) {
      if (observed_if_flag_counts[window.before_task_index] != window.expected_count) {
        return {};
      }
    }

    size_t start_stmt = matched_task_stmt_indices.front();
    size_t end_stmt = matched_task_stmt_indices.back();
    if (HasUnsafeOpenTaskBoundary(body_seq, start_stmt, end_stmt, matched_task_stmt_indices)) {
      return {};
    }

    std::unordered_map<const Call*, RegionCallAttrs> result;
    for (size_t task_index = 0; task_index < matched_task_stmt_indices.size(); ++task_index) {
      auto call = GetDirectCall(body_seq->stmts_[matched_task_stmt_indices[task_index]]);
      INTERNAL_CHECK(call) << "Internal error: matched loop-body task statement has no direct call";
      RegionCallAttrs attrs;
      attrs.template_key = templ.template_key;
      attrs.task_index = static_cast<int>(task_index);
      if (task_index > 0) {
        attrs.dep_indices.push_back(static_cast<int>(task_index - 1));
      }
      result[call.get()] = std::move(attrs);
    }
    return result;
  }
};

}  // namespace

namespace pass {

Pass IdentifyStableRegions() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    if (!program) return program;
    auto new_functions = program->functions_;
    bool changed = false;

    for (auto& [gvar, func] : new_functions) {
      if (!func || !func->body_ || func->func_type_ != FunctionType::Orchestration) {
        continue;
      }
      StableRegionIdentifier identifier;
      auto new_body = identifier.VisitStmt(func->body_);
      if (new_body.get() == func->body_.get()) {
        continue;
      }
      func = std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                        func->return_types_, new_body, func->span_, func->func_type_,
                                        func->level_, func->role_, func->attrs_);
      changed = true;
    }

    if (!changed) {
      return program;
    }
    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "IdentifyStableRegions", kIdentifyStableRegionsProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
