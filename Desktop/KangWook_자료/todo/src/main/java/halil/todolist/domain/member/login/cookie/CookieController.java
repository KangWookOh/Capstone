package halil.todolist.domain.member.login.cookie;

import halil.todolist.domain.member.dto.LoginDto;
import halil.todolist.domain.member.dto.SignUpDto;
import halil.todolist.domain.member.entity.Member;
import halil.todolist.domain.member.service.MemberService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import javax.servlet.http.HttpServletResponse;

@Controller
@RequiredArgsConstructor
public class CookieController {

    private final CookieService cookieService;
    private final MemberService memberService;

    @GetMapping("/cookie/signup")
    public String signUpForm(@ModelAttribute SignUpDto signUpDto) {
        // model.addAttribute("signUpDto", new SignUpDto());
        return "/cookie/signup";
    }

    @PostMapping("/cookie/signup")
    public String signup(@ModelAttribute("signUpDto") SignUpDto signUpDto, Model model) {
        model.addAttribute("signUpDto", memberService.signUp(signUpDto));
        return "redirect:/cookie/login";
    }

    /*
     * GET /Login Form
     */
    @GetMapping("/cookie/login")
    public String loginForm(@ModelAttribute LoginDto loginDto) {
        // model.addAttribute("loginDto", new LoginDto());
        return "/cookie/login";
    }

    @ResponseBody
    @GetMapping("/cookie/get/{id}")
    public ResponseEntity getCookie(@CookieValue(value = "memberId", required = true) @PathVariable("id") Long id) {
        if (id != null) {
            return ResponseEntity.ok(id);
        }

        return ResponseEntity.ok(null);
    }

    /*
     * POST /쿠키 생성과 로그인
     */
    @PostMapping("/cookie/login")
    public String login(@ModelAttribute LoginDto loginDto, HttpServletResponse response) {
        Member member = cookieService.login(response, loginDto.getEmail(), loginDto.getPassword());

        // cookie.getValue() 도 memberId 와 동일
        return String.format("redirect:/cookie/get/%s", member.getId());
    }
}
