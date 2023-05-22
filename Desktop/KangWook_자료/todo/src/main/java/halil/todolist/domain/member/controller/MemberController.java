package halil.todolist.domain.member.controller;


import halil.todolist.domain.member.dto.LoginDto;
import halil.todolist.domain.member.dto.SignUpDto;
import halil.todolist.domain.member.entity.Member;
import halil.todolist.domain.member.service.MemberService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;
import javax.validation.Valid;

@Controller
@RequiredArgsConstructor
public class MemberController {

    private final MemberService memberService;

    @ResponseBody
    @GetMapping("/test")
    public String test() {
        return "test OK";
    }

    /**
     * thymeleaf th:object와 매핑
     * @param model
     * @return
     */
    @GetMapping("/signup")
    public String signUpForm(Model model) {
        model.addAttribute("signUpDto", new SignUpDto());
        return "/signup";
    }

    @PostMapping("/signup")
    public String signup(@Valid @ModelAttribute("signUpDto") SignUpDto signUpDto, Model model) {
        model.addAttribute("signUpDto", memberService.signUp(signUpDto));
        return "redirect:/login";
    }

    @ResponseBody
    @GetMapping("/sign/suc")
    public String signSuccess() {
        return "sign Success";
    }

    @GetMapping("/login")
    public String loginForm(@ModelAttribute LoginDto loginDto) {
        return "/login";
    }

//    @ResponseBody
//    @PostMapping("/login")
//    public ResponseEntity login(@Valid @ModelAttribute LoginDto loginDto) {
//        return ResponseEntity.ok(memberService.login(loginDto.getEmail(), loginDto.getPassword()));
//    }

    @PostMapping("/login")
    public String login(@Valid @ModelAttribute LoginDto loginDto) {
        memberService.login(loginDto.getEmail(), loginDto.getPassword());
        return "redirect:/todos";
    }
}
